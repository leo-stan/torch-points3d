import torch
from dataclasses import dataclass
import ast
from torch.functional import _return_counts

from torch_geometric.data import Dataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId, ShiftVoxels, instantiate_transforms

import copclib as copc
import numpy as np
import os
import json
import glob
from sys import float_info
from omegaconf import OmegaConf
import random
import time
from tqdm import tqdm
from joblib import Parallel, delayed


@dataclass
class DatasetSample:
    file: str
    dataset: str
    depth: int
    x: int
    y: int
    z: list


@dataclass
class File:
    name: str
    hierarchy: dict
    max_depth: int


class CopcInternalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        samples,
        transform,
        train_classes,
        resolution,
        datasets,
        hUnits=1.0,
        vUnits=1.0,
        is_inference=False,
        do_augment=False,
        do_shift=False,
        augment_transform=None,
        train_classes_weights=None,
        min_num_points=1,
    ):

        self.root = root
        self.samples = samples

        self.random_sample = not is_inference and split == "train"

        # Compute total number of samples
        self.sample_probability = None
        if not is_inference:
            if self.random_sample:
                self.nb_samples = sum([dataset["num_training_samples"] for dataset in datasets.values()])

                # the probability any sample will be drawn is equal to the ratio of
                # how many samples we choose for that dataset to the total number of samples
                self.sample_probability = []
                for dset_name, dataset in datasets.items():
                    dset_sampling_probability = dataset["num_training_samples"] / self.nb_samples
                    self.sample_probability.extend([dset_sampling_probability] * len(self.samples[dset_name]))

            else:
                self.nb_samples = sum([len(dset_samples) for dset_samples in self.samples.values()])

            # flatten the list of samples
            self.samples = [item for sublist in self.samples.values() for item in sublist]
        else:
            self.nb_samples = len(self.samples)

        if self.sample_probability is not None:
            # normalize sample probability
            self.sample_probability = np.array(self.sample_probability)
            self.sample_probability = self.sample_probability / np.sum(self.sample_probability)
            assert len(self.samples) == len(self.sample_probability)

        self.is_inference = is_inference
        self.min_num_points = max(1, min_num_points)
        self.resolution = resolution
        self.transform = transform
        self.train_classes = train_classes
        self.hUnits = hUnits
        self.vUnits = vUnits
        self.do_augment = do_augment
        self.do_shift = do_shift
        self.augment_transform = augment_transform
        self.datasets = datasets
        self.retry_idx = 0

        if split == "train":
            self.weight_classes = torch.Tensor(train_classes_weights)

    def __len__(self):
        return self.nb_samples

    # given a key, recursively check if each of its 8 children exist in the hierarchy
    def get_all_key_children(self, key, max_depth, hierarchy, children):
        # stop once we reach max_depth, since none of its children can exist
        if key.d >= max_depth:
            return
        for child in key.GetChildren():
            if str(child) in hierarchy:
                # if the key exists, add it to the output, and check if any of its children exist
                children.append(hierarchy[str(child)])
                self.get_all_key_children(child, max_depth, hierarchy, children)

    def __getitem__(self, idx):
        if self.random_sample:
            # randomly choose a sample
            sample = np.random.choice(self.samples, p=self.sample_probability)
        else:
            sample = self.samples[idx]
        if not self.is_inference:
            dataset = self.datasets[sample.dataset]
            file = dataset["files"][sample.file]
            hierarchy = file.hierarchy

            file_path = os.path.join(self.root,sample.dataset,"copc",sample.file,"octree.copc.laz")
            reader = copc.FileReader(file_path)
            max_depth = file.max_depth
        else:
            hierarchy = self.datasets.hierarchy
            max_depth = self.datasets.max_depth
            reader = copc.FileReader(self.datasets.name)

        header = reader.GetLasHeader()

        # Extract nearest depth, x, and y from sample
        nearest_depth, x, y = sample.depth, sample.x, sample.y
        # Fill z as 0, since we don't care about that dimension
        sample_bounds = copc.Box(copc.VoxelKey(nearest_depth, x, y, 0), header)
        # Make the tile 2D
        sample_bounds.z_min = float_info.min
        sample_bounds.z_max = float_info.max

        # If training we can just grab points without tracking
        valid_nodes = {}
        valid_parent_nodes = {}
        # check each possible z to see if it, or any of its parents, exist
        for i, z in enumerate(sample.z):
            start_key = copc.VoxelKey(nearest_depth, x, y, z)

            # start by checking if the node itself exists
            if str(start_key) in hierarchy:
                # if the node exists, then get all its children
                child_nodes = []
                self.get_all_key_children(start_key, max_depth, hierarchy, child_nodes)
                for child_node in child_nodes:
                    valid_nodes[str(child_node.key)] = child_node
                    
            else:
                # if the node doens't exist, get its first parent to exist
                while str(start_key) not in hierarchy:
                    if start_key.d < 0:
                        raise RuntimeError("This shouldn't happen!")
                    start_key = start_key.GetParent()
            # then, get all nodes from depth 0 to the current depth
            key = start_key
            while key.IsValid():
                valid_parent_nodes[str(key)] = hierarchy[str(key)]
                key = key.GetParent()

        # Process keys that exist
        copc_points = copc.Points(header)
        points_key = []
        points_idx = []
        points_file = []

        # For node and children we can load all points
        for node in valid_nodes.values():
            node_points = reader.GetPoints(node)
            if self.is_inference:
                for i in range(len(node_points)):
                    points_key.append((node.key.d,node.key.x,node.key.y,node.key.z))
                    points_file.append(file_path)
                    points_idx.append(i)
            else:
                copc_points.AddPoints(node_points)

        # For parents node we need to check which points fit within bounds
        for node in valid_parent_nodes.values():
            if self.is_inference:
                node_points = reader.GetPoints(node)
                for i, point in enumerate(node_points):
                    if point.Within(sample_bounds):
                        copc_points.AddPoint(point)
                        points_key.append((node.key.d,node.key.x,node.key.y,node.key.z))
                        points_file.append(file_path)
                        points_idx.append(i)
            copc_points.AddPoints(reader.GetPoints(node).GetWithin(sample_bounds))

        points = np.stack([copc_points.X, copc_points.Y, copc_points.Z], axis=1) # Nx3
        points_key = np.asarray(points_key)  # N
        points_idx = np.asarray(points_idx)  # N
        points_file = np.asarray(points_file)  # N

        classification = np.array(copc_points.Classification).astype(np.int)
        y = torch.from_numpy(classification)

        # we need this check because numpy will error for empty array
        if len(points) >= self.min_num_points:
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

            for filter in dataset["filter_classes"]:
                mask = y != filter
                y = y[mask]
                points = points[mask]
            y = self._remap_labels(y, dataset)

        data = Data(pos=torch.from_numpy(points).type(torch.float), y=y, points_key=points_key, points_idx=points_idx, points_file=points_file)

        if len(data.pos) < self.min_num_points:
            if self.random_sample:
                # if there's no points in this sample, just get another sample:
                return self[0]
            else:
                # there's not really a great way to handle this
                return self[random.randint(0, self.nb_samples-1)]

        if self.is_inference:
            data = SaveOriginalPosId()(data)

        if self.do_augment:
            data = self.augment_transform(data)

        if self.transform:
            data = self.transform(data)

        if self.do_shift:
            data = ShiftVoxels()(data)

        return data

    def _remap_labels(self, labels, dataset):
        NUM_CLASSES = 256  # arbitrary
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = labels.clone()

        # first map using the class_map
        mapping_dict = {f: t for f, t in dataset["class_map_from_to"]}
        for idx in range(NUM_CLASSES):
            if idx not in mapping_dict:
                mapping_dict[idx] = 0

        # add identity mappings for each train class
        for c in self.train_classes:
            mapping_dict[c] = c

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
        return self[0].num_node_features

    @property
    def num_classes(self):
        return len(self.train_classes) + 1


def get_hierarchy(file, file_path, resolution):
    reader = copc.FileReader(file_path)
    max_depth = reader.GetDepthAtResolution(resolution)
    hierarchy = {str(node.key): node for node in reader.GetAllChildren() if node.key.d <= max_depth}
    return File(file, hierarchy, max_depth)


class CopcDatasetFactory(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        splits = {"train": {}, "test": {}, "val": {}}
        datasets = OmegaConf.to_container(dataset_opt.datasets, True)

        for dset_name, dataset in datasets.items():
            with open(
                os.path.join(
                    dataset_opt.dataroot, dset_name, "copc/splits-v%d.json" % (dataset_opt.dataset_version)
                )
            ) as fp:
                dset_splits = json.load(fp)

            dataset["file_list"] = set()
            for split in splits.keys():
                splits[split][dset_name] = []

                if split == "train" and dataset["num_training_samples"] == 0:
                    continue

                for file_name, file_items in dset_splits[split].items():
                    dataset["file_list"].add(file_name)
                    for dxy, z_list in file_items.items():
                        d, x, y = ast.literal_eval(dxy)
                        splits[split][dset_name].append(
                            DatasetSample(file=file_name, dataset=dset_name, depth=d, x=x, y=y, z=z_list)
                        )

                if split == "train" and dataset["num_training_samples"] < 0:
                    dataset["num_training_samples"] = len(splits[split][dset_name])

            out_files = Parallel(n_jobs=1)(delayed(get_hierarchy)(file, os.path.join(dataset_opt.dataroot, dset_name, "copc", file, "octree.copc.laz"), dataset_opt.resolution) for file in tqdm(dataset["file_list"]))
            dataset["files"] = {file.name: file for file in out_files}

        self.train_dataset = CopcInternalDataset(
            root=dataset_opt.dataroot,
            split="train",
            samples=splits["train"],
            transform=self.train_transform,
            train_classes=dataset_opt.training_classes,
            resolution=dataset_opt.resolution,
            min_num_points=dataset_opt.min_num_points,
            datasets=datasets,
            do_augment=dataset_opt.do_augment,
            do_shift=dataset_opt.do_shift,
            # augment_transform=instantiate_transforms(dataset_opt.augment),
            train_classes_weights=dataset_opt.training_classes_weights,
        )

        self.val_dataset = CopcInternalDataset(
            root=dataset_opt.dataroot,
            split="val",
            samples=splits["val"],
            datasets=datasets,
            train_classes=dataset_opt.training_classes,
            transform=self.val_transform,
            resolution=dataset_opt.resolution,
            min_num_points=dataset_opt.min_num_points,
            do_augment=False,
            do_shift=dataset_opt.do_shift,
        )

        self.test_dataset = CopcInternalDataset(
            root=dataset_opt.dataroot,
            split="test",
            samples=splits["test"],
            datasets=datasets,
            train_classes=dataset_opt.training_classes,
            transform=self.test_transform,
            resolution=dataset_opt.resolution,
            min_num_points=dataset_opt.min_num_points,
            do_augment=False,
            do_shift=dataset_opt.do_shift,
        )


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


class CopcDatasetFactoryInference(BaseDataset):
    def __init__(self, dataset_opt, keys):
        super().__init__(dataset_opt)

        dataset = get_hierarchy(dataset_opt.inference_file, file_path=dataset_opt.inference_file,
                                resolution=dataset_opt.resolution)

        splits = []

        for dxy, z_list in keys.items():
            d, x, y = ast.literal_eval(dxy)
            splits.append(DatasetSample(file="", dataset="", depth=d, x=x, y=y, z=z_list))

        self.test_dataset = CopcInternalDataset(
            root=dataset_opt.dataroot,
            split="inference",
            samples=splits,
            datasets=dataset,
            train_classes=dataset_opt.training_classes,
            is_inference=True,
            transform=self.test_transform,
            resolution=dataset_opt.resolution,
            min_num_points=dataset_opt.min_num_points,
            do_augment=False,
            do_shift=dataset_opt.do_shift
        )