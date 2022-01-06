import torch
from dataclasses import dataclass
import ast
from torch_geometric.data import Data
import copclib as copc
import numpy as np
import os
import json
from omegaconf import OmegaConf
import math
from tqdm import tqdm
from joblib import Parallel, delayed
import random

from torch_points3d.core.data_transform.grid_transform import GridSampling3D

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import ShiftVoxels, AddFeatsByKeys
from torch_points3d.core.data_transform import instantiate_transforms
from torch_points3d.core.data_transform import SaveOriginalPosId

from utils.dataset_helpers import get_all_points, get_valid_nodes, get_sample_bounds


@dataclass
class DatasetSample:
    file: str
    dataset: str
    depth: int
    x: int
    y: int
    z: list
    label_hist: dict


@dataclass
class File:
    name: str
    path: str
    hierarchy: dict
    max_depth: int


class CopcInternalDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset_options,
        split,
        samples,
        transform,
        datasets,
        hUnits=1.0,
        vUnits=1.0,
    ):
        super().__init__()

        self.root = dataset_options.dataroot
        self.is_inference = split == "inference"
        self.datasets = datasets
        self.dataset_options = dataset_options

        self.random_sample = not self.is_inference and split == "train"

        # Create remapping dictionary
        self.remapping_dict = {}
        for dset_name, dataset in datasets.items():
            self.remapping_dict[dset_name] = create_remap_dict(dataset["class_map_from_to"], dataset_options.training_classes)

        self.sample_probability = None
        if self.random_sample:
            # Compute total requested number of samples
            self.num_samples = sum([dataset["num_training_samples"] for dataset in datasets.values()])
            # Compute the probabilities of samples based on requested ratios
            self.compute_sample_probabilities(samples)
        else:
            # Compute total number of samples
            self.num_samples = sum([len(dset_samples) for dset_samples in samples.values()])

        # flatten the list of samples
        self.samples = [item for sublist in samples.values() for item in sublist]
        if self.sample_probability:
            # normalize sample probability
            self.sample_probability = np.array(self.sample_probability)
            self.sample_probability = self.sample_probability / np.sum(self.sample_probability)
            assert len(self.samples) == len(self.sample_probability)

        self.min_num_points = max(1, dataset_options.min_num_points)
        self.resolution = dataset_options.resolution
        self.transform = transform
        self.train_classes = dataset_options.training_classes
        self.hUnits = hUnits
        self.vUnits = vUnits
        self.do_shift = dataset_options.get("do_shift", False)
        self.rgb_norm = dataset_options.get("rgb_norm", None)

        self.start_idx = None
        self.end_idx = None

        self.augment_transform = None
        self.include_other_samples_prob = None
        if split == "train":
            if dataset_options.get("augment_transform", None):
                self.augment_transform = instantiate_transforms(dataset_options.augment_transform)
            if dataset_options.training_classes_weights is not None:
                self.weight_classes = torch.Tensor(dataset_options.training_classes_weights)
            self.include_other_samples_prob = dataset_options.get("include_other_samples_prob", 0)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.start_idx is None or self.end_idx is None:
            raise RuntimeError("this shouldn't happen!")

        for idx in range(self.start_idx, self.end_idx):
            data = self.get_sample(idx)
            if data is None:
                continue
            else:
                yield data

    def __getitem__(self, idx):
        return self.get_sample(idx)

    def get_sample(self, idx):
        if self.random_sample:
            # randomly choose a sample
            sample = np.random.choice(self.samples, p=self.sample_probability)
        else:
            sample = self.samples[idx]

        # If we have label histogram data, compute sample histogram
        sample_hist = {}
        if len(sample.label_hist) > 0:
            for label, count in sample.label_hist.items():
                sample_hist[self.remapping_dict[sample.dataset][int(label)]] = count

        # shortcuts
        dataset = self.datasets[sample.dataset]
        file = dataset["files"][sample.file]
        hierarchy = file.hierarchy

        # create reader
        reader = copc.FileReader(file.path)
        header = reader.copc_config.las_header
        max_depth = file.max_depth

        # Extract nearest depth, x, and y from sample
        nearest_depth, x, y = sample.depth, sample.x, sample.y
        # make a 2d box out of the voxelkey x,y
        sample_bounds = get_sample_bounds(nearest_depth, x, y, header)

        valid_child_nodes, valid_parent_nodes = get_valid_nodes(sample.z, hierarchy, max_depth, nearest_depth, x, y)

        copc_points, points_key, points_idx = get_all_points(
            reader, header, sample_bounds, valid_child_nodes, valid_parent_nodes, self.is_inference
        )

        if len(copc_points) < self.min_num_points:
            return None

        points = np.stack([copc_points.x, copc_points.y, copc_points.z], axis=1)
        rgb = np.stack([copc_points.red, copc_points.green, copc_points.blue], axis=1).astype(float)
        y = np.array(copc_points.classification).astype(np.int)

        if self.rgb_norm is not None:
            rgb = self.normalize_rgb(reader, rgb)

        self.normalize_points(points)

        for filter in dataset["filter_classes"]:
            # don't keep filtering if all the points got filtered out
            if len(points) < self.min_num_points:
                break

            mask = y != filter
            y = y[mask]
            points = points[mask]
            if self.rgb_norm is not None:
                rgb = rgb[mask]

        if len(points) < self.min_num_points:
            return None

        y = self._remap_labels(torch.from_numpy(y), dataset)
        # check if we actually have any training classes
        if self.random_sample and not any([label > 0 for label in np.unique(y)]):
            if random.random() > self.include_other_samples_prob:
                return None

        data = Data(pos=torch.from_numpy(points).type(torch.float), y=y, label_hist=sample_hist)
        if self.rgb_norm is not None:
            data.rgb = torch.from_numpy(rgb).type(torch.float)
        if self.is_inference:
            points_key = np.concatenate(points_key)
            points_idx = np.concatenate(points_idx)
            pos_orig = torch.from_numpy(points.copy()).type(torch.float)

            data.points_key = torch.from_numpy(np.asarray(points_key))
            data.points_idx = torch.from_numpy(np.asarray(points_idx))
            data.pos_orig = pos_orig

            grid_start = torch.round(torch.Tensor([header.min.x, header.min.y, header.min.z]) / self.resolution)
            data.grid_start = grid_start

            data = SaveOriginalPosId()(data)
            data = GridSampling3D(
                self.resolution,
                quantize_coords=True,
                mode="last_noshuffle",
                skip_keys=["points_key", "points_idx", "pos_orig"],
                grid_start=grid_start,
            )(data)
            self.transform.transforms = [x for x in self.transform.transforms if not isinstance(x, GridSampling3D)]

        if self.augment_transform:
            data = self.augment_transform(data)

        if self.transform:
            data = self.transform(data)

        if self.do_shift:
            data = ShiftVoxels()(data)
        if len(data.pos) == 0:
            raise RuntimeError()

        return data

    def compute_sample_probabilities(self, samples):
        num_total_samples = sum([len(samples[dset_name]) for dset_name in self.datasets.keys()])

        # We apply a correction factor based on the dataset size compared to the others
        # and the requested ratio compared to the others
        self.sample_probability = []
        # Go through each dataset to compute the adjustment factor
        for dset_name, dataset in self.datasets.items():
            # Compute the ratio of number of samples for this dataset over the total number of samples of all dsets
            native_ratio = len(samples[dset_name]) / num_total_samples
            # Compute the ratio of requested number of training samples for this datasets over the total number of
            # requested samples
            requested_ratio = dataset["num_training_samples"] / self.num_samples
            # The sampling factor adjusts the sampling rate based on the size of the current dataset compared to
            # others, and the requested ratio of samples from this dataset compared to others
            dset_sampling_factor = requested_ratio / native_ratio
            # We assign the sampling factor of this dataset to all its samples
            self.sample_probability.extend([dset_sampling_factor] * len(samples[dset_name]))

    def normalize_points(self, points):
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        xoff = (int(x_min) + int(x_max)) // 2
        yoff = (int(y_min) + int(y_max)) // 2
        points[:, 0] -= np.round(xoff).astype(int)
        points[:, 1] -= np.round(yoff).astype(int)

        # Z centering using mean
        z_mean = np.mean(points[:, 2])
        points[:, 2] -= int(z_mean)

        points[:, :2] *= self.hUnits
        points[:, 2] *= self.vUnits

    def normalize_rgb(self, reader, rgb):
        extents = reader.copc_config.copc_extents
        max_rgb = max(extents.red.maximum, extents.green.maximum, extents.blue.maximum)

        if max_rgb > 0:
            if max_rgb > 255:
                norm_factor = 65535
            else:
                norm_factor = 255

            rgb = rgb / norm_factor
            rgb = (rgb - np.array(self.rgb_norm.mean)) / np.sqrt(np.array(self.rgb_norm.var_s))
        return rgb

    def _remap_labels(self, labels, dataset):
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = labels.clone()

        mapping_dict = create_remap_dict(dataset["class_map_from_to"], self.train_classes)

        # now shift the existing classes so that they are consecutive
        for i, c in enumerate(self.train_classes):
            for c1, c2 in mapping_dict.items():
                # the existing mapping list maps from one class to another class
                # so we take that second class and check if we want to keep it, i.e. it's in train_classes
                # and if so, we set it to 1+i because i is 0-based and 0 is set to the catch-all class, so we start indexing at 1
                if c2 == c:
                    mapping_dict[c1] = i + 1

        for idx in dataset["ignore_classes"]:
            mapping_dict[idx] = IGNORE_LABEL

        for source, target in mapping_dict.items():
            mask = labels == source
            new_labels[mask] = target

        return new_labels

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        # count number of features in advance
        feat_transform = [x for x in self.transform.transforms if isinstance(x, AddFeatsByKeys)][0]

        num_feats = 0
        for feat_name in feat_transform._feat_names:
            if feat_name.startswith("pos_"):
                num_feats += 1
            elif feat_name == "rgb":
                num_feats += 3
            else:
                num_feats += 1

        return num_feats

    @property
    def num_classes(self):
        return len(self.train_classes) + 1


def get_hierarchy(file, file_path, resolution):
    reader = copc.FileReader(file_path)
    max_depth = reader.GetDepthAtResolution(resolution)
    hierarchy = {str(node.key): node for node in reader.GetAllNodes() if node.key.d <= max_depth}
    return File(file, file_path, hierarchy, max_depth)


def create_remap_dict(class_map_from_to, train_classes):
    NUM_CLASSES = 256  # arbitrary

    # first map using the class_map
    mapping_dict = {f: t for f, t in class_map_from_to}
    for idx in range(NUM_CLASSES):
        if idx not in mapping_dict:
            mapping_dict[idx] = 0

    # add identity mappings for each train class
    for c in train_classes:
        mapping_dict[c] = c
    return mapping_dict


class CopcBaseDatasetFactory(BaseDataset):
    def __init__(self, dataset_opt, internal_dataset):
        super().__init__(dataset_opt)

        splits = {"train": {}, "test": {}, "val": {}}
        datasets = OmegaConf.to_container(dataset_opt.datasets, resolve=True)

        # parse through each dataset's splits file and merge them
        for dset_name, dataset in datasets.items():
            with open(
                os.path.join(dataset_opt.dataroot, dset_name, "splits-v%d.json" % (dataset_opt.dataset_version))
            ) as fp:
                dset_splits = json.load(fp)

            dataset["file_list"] = set()
            for split in splits.keys():
                splits[split][dset_name] = []

                if split == "train" and dataset["num_training_samples"] == 0:
                    continue

                for file_name, file_items in dset_splits[split].items():
                    dataset["file_list"].add(file_name)
                    for dxy, values in file_items.items():
                        # tuple string to tuple
                        d, x, y = ast.literal_eval(dxy)
                        splits[split][dset_name].append(
                            DatasetSample(file=file_name, dataset=dset_name, depth=d, x=x, y=y, z=values["z"], label_hist=values["class_hist"])
                        )

                if split == "train" and dataset["num_training_samples"] < 0:
                    dataset["num_training_samples"] = len(splits[split][dset_name])

            out_files = Parallel(n_jobs=-1)(
                delayed(get_hierarchy)(
                    file,
                    os.path.join(dataset_opt.dataroot, dset_name, file, "octree.copc.laz"),
                    dataset_opt.resolution,
                )
                for file in tqdm(dataset["file_list"])
            )
            dataset["files"] = {file.name: file for file in out_files}

        self.train_dataset = internal_dataset(
            dataset_options=dataset_opt,
            split="train",
            samples=splits["train"],
            transform=self.train_transform,
            datasets=datasets,
        )

        self.val_dataset = internal_dataset(
            dataset_options=dataset_opt,
            split="val",
            samples=splits["val"],
            datasets=datasets,
            transform=self.val_transform,
        )

        self.test_dataset = internal_dataset(
            dataset_options=dataset_opt,
            split="test",
            samples=splits["test"],
            datasets=datasets,
            transform=self.test_transform,
        )

    # override this so it's not constantly asking for data objects
    def has_labels(self, stage: str) -> bool:
        return True

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

        return SegmentationTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=IGNORE_LABEL
        )

    def worker_init_fn(self, worker_id):
        do_worker_init(worker_id)


def do_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = len(dataset)
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    dataset.start_idx = overall_start + worker_id * per_worker
    dataset.end_idx = min(dataset.start_idx + per_worker, overall_end)


class CopcDatasetFactory(CopcBaseDatasetFactory):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt, CopcInternalDataset)


class CopcDatasetFactoryInference(BaseDataset):
    def __init__(self, dataset_opt, keys, hUnits, vUnits, min_resolution):
        super().__init__(dataset_opt)

        file = get_hierarchy(
            dataset_opt.inference_file, file_path=dataset_opt.inference_file, resolution=min_resolution
        )

        datasets = {
            "inference_dset": {
                "files": {"inference_file": file},
                "ignore_classes": [],
                "filter_classes": [],
                "class_map_from_to": [],
            }
        }
        splits = {"inference_dset": []}

        for dxy, z_list in keys.items():
            d, x, y = ast.literal_eval(dxy)
            splits["inference_dset"].append(
                DatasetSample(file="inference_file", dataset="inference_dset", depth=d, x=x, y=y, z=z_list, label_hist={})
            )

        self.test_dataset = CopcInternalDataset(
            dataset_options=dataset_opt,
            split="inference",
            samples=splits,
            datasets=datasets,
            transform=self.test_transform,
            hUnits=hUnits,
            vUnits=vUnits,
        )

    # override this so it's not constantly asking for data objects
    def has_labels(self, stage: str) -> bool:
        return False

    def worker_init_fn(self, worker_id):
        do_worker_init(worker_id)
