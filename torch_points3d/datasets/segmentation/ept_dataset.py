from dataclasses import dataclass

from torch_geometric.data import Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation import IGNORE_LABEL

import torch
import laspy
import numpy as np
import os
import json


@dataclass
class DataFile:
    files: list
    mins: list
    maxs: list
    mid: list
    dset_idx: int


class EptInternalDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, datasets, num_classes):

        self.files = []
        for i, dset in enumerate(datasets):
            splits_path = os.path.join(root, dset.dataset, "splits.json")
            with open(splits_path) as f:
                splits = json.load(f)

            split_files = splits[split]

            metadataPath = os.path.join(root, dset.dataset, "ept.json")
            with open(metadataPath) as f:
                metadata = json.load(f)
            bounds = metadata["bounds"]

            filePath = os.path.join(root, dset.dataset, "ept-data")
            files = [
                self._get_hierarchy_files(bounds, x, filePath, i) for x in split_files
            ]

            self.files.extend(files)
            dset.offsets = [
                metadata["schema"][0]["offset"],
                metadata["schema"][1]["offset"],
                metadata["schema"][2]["offset"],
            ]
            dset.scales = [
                metadata["schema"][0]["scale"],
                metadata["schema"][1]["scale"],
                metadata["schema"][2]["scale"],
            ]

        self.root = root
        self.datasets = datasets
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.files)

    def _get_hierarchy_files(self, bounds, file, path, dset_idx):
        cubeSize = bounds[3] - bounds[0]
        depth, x, y, z = [int(i) for i in file.split("-")]
        nodes = [file]

        dx, dy, dz = x, y, z
        for currentDepth in reversed(range(depth)):
            dx, dy, dz = dx // 2, dy // 2, dz // 2

            fname = "{}-{}-{}-{}".format(currentDepth, dx, dy, dz)
            nodes.append(fname)

        minx, miny, minz = bounds[:3]
        currentSpan = cubeSize / pow(2, depth)

        minx = minx + x * currentSpan
        miny = miny + y * currentSpan
        minz = minz + z * currentSpan

        maxx, maxy, maxz = minx + currentSpan, miny + currentSpan, minz + currentSpan

        mins = [minx, miny, minz]
        maxs = [maxx, maxy, maxz]
        mid = np.round((np.array(maxs) - np.array(mins)) / 2)
        mid += mins
        files = [os.path.join(path, x + ".laz") for x in nodes]
        return DataFile(files, mins, maxs, mid, dset_idx)

    def __getitem__(self, idx):
        datafile = self.files[idx]
        dset = self.datasets[datafile.dset_idx]

        all_points = None
        all_classes = None
        for file in datafile.files:
            las = laspy.read(file)
            points = np.stack([las.X, las.Y, las.Z], axis=1)
            classification = las.classification.astype(np.int)

            # clip the points to the lowest depth bounding box
            points_float = np.stack([las.x, las.y, las.z], axis=1)
            in_min = np.all(points_float >= datafile.mins, axis=1)
            in_max = np.all(points_float <= datafile.maxs, axis=1)
            valid_points = np.logical_and(in_min, in_max)

            # filter out ignored point classifications
            valid_classes = ~np.in1d(classification, dset.filter_classes)
            valid_points = np.logical_and(valid_points, valid_classes)

            # filter
            points = points[valid_points]
            classification = classification[valid_points]

            # handle ignore_classes and train_classes
            class_map = {frm: to for (frm, to) in zip(dset.class_map_from, dset.class_map_to)}
            classification = self._remap_labels(classification, dset.ignore_classes, dset.train_classes, class_map)
            
            points = points.astype(np.float64)
            points[:, 0] *= dset.scales[0]
            points[:, 1] *= dset.scales[1]
            points[:, 2] *= dset.scales[2]

            if all_points is None:
                all_points = points
                all_classes = classification
            else:
                all_points = np.concatenate((all_points, points), axis=0)
                all_classes = np.concatenate((all_classes, classification))
            print(all_points.shape)
            
        all_points[:, 0] -= all_points[:, 0].min()
        all_points[:, 1] -= all_points[:, 1].min()
        all_points[:, 0] -= all_points[:, 0].max() / 2
        all_points[:, 1] -= all_points[:, 1].max() / 2
        all_points[:, 2] -= all_points[:, 2].mean()

        data = Data(pos=torch.from_numpy(all_points).type(torch.float), y=all_classes)
        if self.transform:
            data = self.transform(data)

        return data

    def _remap_labels(self, labels, ignore_classes, train_classes, class_map):
        NUM_CLASSES = 100  # arbitrary
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = torch.from_numpy(labels).clone()

        # first map using the class_map
        mapping_dict = {k: v for (k, v) in class_map.items()}
        for idx in range(NUM_CLASSES):
            if idx not in mapping_dict:
                mapping_dict[idx] = 0

        # add identity mappings for each train class
        for c in train_classes:
            mapping_dict[c] = c

        # now shift the existing classes so that they are consecutive
        for i, c in enumerate(train_classes):
            for c1, c2 in mapping_dict.items():
                # the existing mapping list maps from one class to another class
                # so we take that second class and check if we want to keep it, i.e. it's in train_classes
                # and if so, we set it to 1+i because i is 0-based and 0 is set to the catch-all class, so we start indexing at 1
                if c2 == c:
                    mapping_dict[c1] = i + 1

        for idx in ignore_classes:
            mapping_dict[idx] = IGNORE_LABEL

        for source, target in mapping_dict.items():
            mask = labels == source
            new_labels[mask] = target

        return new_labels

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self[0].num_node_features


class EptDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = EptInternalDataset(
            dataset_opt.dataroot,
            split="train",
            transform=self.train_transform,
            datasets=dataset_opt.datasets,
            num_classes=dataset_opt.num_classes,
        )

        self.val_dataset = EptInternalDataset(
            dataset_opt.dataroot,
            split="val",
            transform=self.train_transform,
            datasets=dataset_opt.datasets,
            num_classes=dataset_opt.num_classes,
        )

        self.test_dataset = EptInternalDataset(
            dataset_opt.dataroot,
            split="test",
            transform=self.train_transform,
            datasets=dataset_opt.datasets,
            num_classes=dataset_opt.num_classes,
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
            self,
            wandb_log=wandb_log,
            use_tensorboard=tensorboard_log,
            ignore_label=IGNORE_LABEL,
        )
