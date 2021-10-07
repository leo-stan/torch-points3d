import torch
from dataclasses import dataclass
import ast

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
import re

@dataclass
class DatasetSample:
    file: str
    depth: int
    x: int
    y: int
    z: list

class CopcInternalDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, samples, transform, train_classes, resolution, datasets, hUnits=1.0, vUnits=1.0, number_of_sample_per_epoch=-1, is_inference=False, do_augment=False, do_shift=False, augment_transform=None, train_classes_weights=None):

        self.root = root
        self.samples = samples

        # # TODO: Compute nb samples per dataset and total number of samples
        # [dataset,split,bound]
        # for dataset in samples.keys():

        # Compute total number of samples
        if number_of_sample_per_epoch >= 0:
            self.nb_samples = number_of_sample_per_epoch
        else:
            self.nb_samples = sum([len(dset_samples) for dset_samples in self.samples.values()])

        dataset_sampling_rates = [x.sampling_rate for x in datasets.values()]
        self.dataset_sampling_rates = dataset_sampling_rates/np.sum(dataset_sampling_rates) # probably of drawing samples from each dataset, sum to 1

        self.is_inference = is_inference
        self.resolution = resolution
        self.transform = transform
        self.train_classes = train_classes
        self.hUnits = hUnits
        self.vUnits = vUnits
        self.do_augment = do_augment
        self.do_shift = do_shift
        self.augment_transform = augment_transform
        self.datasets = datasets
        
        if split == "train":
            self.weight_classes = torch.Tensor(train_classes_weights)

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):

        # randomly choose a dataset
        dset_name = np.random.choice(list(self.samples.keys()), p=self.dataset_sampling_rates)
        # randomly choose a sample
        sample = np.random.choice(list(self.samples[dset_name]))
        dataset = self.datasets[dset_name]

        reader = copc.FileReader(os.path.join(self.root,dset_name,"copc",sample.file,"octree.copc.laz"))
        header = reader.GetLasHeader()

        # Extract nearest depth, x, and y from sample
        nearest_depth, x, y = sample.depth, sample.x, sample.y
        # Fill z as 0, since we don't care about that dimension
        sample_bounds = copc.Box(copc.VoxelKey(nearest_depth,x,y,0), header)
        # Make the tile 2D
        sample_bounds.z_min = float_info.min
        sample_bounds.z_max = float_info.max
        max_depth = reader.GetDepthAtResolution(self.resolution)

        # Test possible Zs to see if key exist
        valid_keys = set()
        for z in sample.z:
            key = copc.VoxelKey(nearest_depth,x,y,z)
            node = reader.FindNode(key)

            while not node.IsValid() and key.IsValid():
                key = key.GetParent()
                node = reader.FindNode(key)
            if node.IsValid():
                valid_keys.add(key)

        # Process keys that exist
        copc_points = copc.Points(header)
        points_key = []
        points_idx = []
        points_file = []
        loaded_keys = set()

        for key in valid_keys:
            if key.d == nearest_depth:
                # Get all children points (these will automatically fit within sample_bounds)
                for node in reader.GetAllChildren(key):
                    if node.key.depth <= max_depth:
                        node_points = reader.GetPoints(node)
                        copc_points.AddPoints(node_points)
                        if self.is_inference:
                            for i in range(len(node_points)):
                                points_key.append((node.key.d,node.key.x,node.key.y,node.key.z))
                                points_file.append(os.path.join(self.root,dset_name,"copc",sample.file,"octree.copc.laz"))
                                points_idx.append(i)

            while key.IsValid() and key not in loaded_keys:
                if self.is_inference:
                    node_points = reader.GetPoints(key)
                    for i,point in enumerate(node_points):
                        if point.Within(sample_bounds):
                            copc_points.AddPoint(point)
                            points_key.append((key.d,key.x,key.y,key.z))
                            points_file.append(os.path.join(self.root,dset_name,"copc",sample.file,"octree.copc.laz"))
                            points_idx.append(i)
                else:
                    copc_points.AddPoints(reader.GetPoints(key).GetWithin(sample_bounds))
                loaded_keys.add(key)
                key = key.GetParent()

        points = np.stack([copc_points.X, copc_points.Y, copc_points.Z], axis=1) # Nx3
        points_key = np.asarray(points_key)  # N
        points_idx = np.asarray(points_idx)  # N
        points_file = np.asarray(points_file)  # N

        classification = np.array(copc_points.Classification).astype(np.int)

        if len(points) == 0:
            # if there's no points in this sample, just get another sample:
            return self[0]
        
        x_min = np.min(points[:,0])
        y_min = np.min(points[:,1])
        x_max = np.max(points[:,0])
        y_max = np.max(points[:,1])

        xoff = (int(x_min) + int(x_max))//2
        yoff = (int(y_min) + int(y_max))//2
        points[:,0] -= np.round(xoff).astype(int)
        points[:,1] -= np.round(yoff).astype(int)

        # Z centering using mean
        z_mean = np.mean(points[:, 2])
        points[:,2] -= int(z_mean)

        points[:,:2] *= self.hUnits
        points[:,2] *= self.vUnits

        y = torch.from_numpy(classification)
        for filter in dataset.filter_classes:
            mask = y != filter
            y = y[mask]
            points = points[mask]
        y = self._remap_labels(y, dataset)

        data = Data(pos=torch.from_numpy(points).type(torch.float), y=y, points_key=points_key, points_idx=points_idx, points_file=points_file)

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
        NUM_CLASSES = 256 # arbitrary
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = labels.clone()

        # first map using the class_map
        mapping_dict = {f: t for f, t in dataset.class_map_from_to}
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

        for idx in dataset.ignore_classes:
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


class CopcDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        splits = {"train": {}, "test": {}, "val": {}}
        for dset_name, dataset in dataset_opt.datasets.items():
            with open(os.path.join(dataset_opt.dataroot, dset_name, "copc/splits-v%d.json" % (dataset_opt.dataset_version))) as fp:
                dset_splits = json.load(fp)

            for split in splits.keys():
                splits[split][dset_name] = []
                for file_name, file_items in dset_splits[split].items():
                    for dxy, z_list in file_items.items():
                        d, x, y  = ast.literal_eval(dxy)
                        splits[split][dset_name].append(DatasetSample(file=file_name, depth=d, x=x, y=y, z=z_list))

        if not dataset_opt.is_inference:
            self.train_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                split="train",
                samples=splits["train"],
                transform=self.train_transform,
                train_classes=dataset_opt.training_classes,
                resolution=dataset_opt.resolution,
                datasets=dataset_opt.datasets,
                do_augment=dataset_opt.do_augment,
                do_shift=dataset_opt.do_shift,
                number_of_sample_per_epoch=dataset_opt.number_of_sample_per_epoch,
                # augment_transform=instantiate_transforms(dataset_opt.augment),
                train_classes_weights=dataset_opt.training_classes_weights,)

            self.val_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                split="val",
                samples=splits["val"],
                datasets=dataset_opt.datasets,
                number_of_sample_per_epoch=-1,
                train_classes=dataset_opt.training_classes,
                transform=self.val_transform,
                resolution=dataset_opt.resolution,
                do_augment=False,
                do_shift=dataset_opt.do_shift
            )

            self.test_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                split="test",
                samples=splits["test"],
                datasets=dataset_opt.datasets,
                train_classes=dataset_opt.training_classes,
                number_of_sample_per_epoch=-1,
                transform=self.test_transform,
                resolution=dataset_opt.resolution,
                do_augment=False,
                do_shift=dataset_opt.do_shift
            )
        else:

            self.test_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                split="inference",
                samples=splits["test"],
                datasets=dataset_opt.datasets,
                train_classes=dataset_opt.training_classes,
                number_of_sample_per_epoch=-1,
                is_inference=True,
                transform=self.test_transform,
                resolution=dataset_opt.resolution,
                do_augment=False,
                do_shift=dataset_opt.do_shift
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