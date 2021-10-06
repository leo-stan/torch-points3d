import torch

from torch_geometric.data import Dataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId, ShiftVoxels, instantiate_transforms

import copclib as copc
import numpy as np
import os
import json
import glob


class CopcInternalDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, train_classes, files, class_map={}, is_inference=False, donotcare_class_ids=[], do_augment=False, do_shift=False, augment_transform=None, z_centering="min", weight_classes=None):
        self.files = files

        self.is_inference = is_inference

        self.transform = transform
        self.train_classes = train_classes
        self.class_map = dict(class_map)
        self.donotcare_class_ids = donotcare_class_ids
        self.hunits = 1.0
        self.vunits = 1.0
        self.do_augment = do_augment
        self.do_shift = do_shift
        self.augment_transform = augment_transform
        self.z_centering = z_centering
        self.weight_classes = torch.Tensor(weight_classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        reader = copc.Reader(file)
        copc_points = reader.GetAllPoints()

        points = np.stack([copc_points.X, copc_points.Y, copc_points.Z], axis=1)

        classification = np.array(copc_points.Classification).astype(np.int)
        y = torch.from_numpy(classification)
        y = self._remap_labels(y)

        xoff = (int(copc_points.X.min()) + int(copc_points.X.max()))/2
        yoff = (int(copc_points.Y.min()) + int(copc_points.Y.max()))/2
        points[:,0] -= np.round(xoff).astype(int)
        points[:,1] -= np.round(yoff).astype(int)

        if self.z_centering == "mean":
            points[:,2] -= int(copc_points.Z.mean())
        elif self.z_centering == "min":
            min = copc_points.Z[classification!=7].min()
            points[:,2] -= min
        else:
            print("Warning! No z_centering recognized")

        points = points.astype(np.float64)
        header = reader.GetLasHeader()
        points[:,0] *= self.hunits * header.scale.x
        points[:,1] *= self.hunits * header.scale.y
        points[:,2] *= self.vunits * header.scale.z

        data = Data(pos=torch.from_numpy(points).type(torch.float), y=y, file=[idx])

        if self.is_inference:
            data = SaveOriginalPosId()(data)

        if self.do_augment:
            data = self.augment_transform(data)

        if self.transform:
            data = self.transform(data)

        if self.do_shift:
            data = ShiftVoxels()(data)

        return data

    def _remap_labels(self, labels):
        NUM_CLASSES = 100 # arbitrary
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = labels.clone()

        # first map using the class_map
        mapping_dict = {k:v for (k, v) in self.class_map.items()}
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

        for idx in self.donotcare_class_ids:
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

        if not dataset_opt.is_test:
            class_map = dict(dataset_opt.classification_map.class_map_from_to)
            with open(os.path.join(dataset_opt.dataroot, "splits_v%d.json" % (dataset_opt.dataset_version)), 'r') as fp:
                splits = json.load(fp)

            self.train_dataset = CopcInternalDataset(
                os.path.join(dataset_opt.dataroot, 'tiles'),
                split="train",
                transform=self.train_transform,
                train_classes=dataset_opt.train_classes,
                class_map=class_map,
                files=[os.path.join(dataset_opt.dataroot, 'tiles', x) for x in splits["train"]],
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=dataset_opt.do_augment,
                do_shift=dataset_opt.do_shift,
                augment_transform=instantiate_transforms(dataset_opt.augment),
                z_centering=dataset_opt.z_centering,
                weight_classes=dataset_opt.weight_classes,
            )

            self.val_dataset = CopcInternalDataset(
                os.path.join(dataset_opt.dataroot, 'tiles'),
                split="val",
                transform=self.val_transform,
                train_classes=dataset_opt.train_classes,
                class_map=class_map,
                files=[os.path.join(dataset_opt.dataroot, 'tiles', x) for x in splits["val"]],
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=False,
                do_shift=dataset_opt.do_shift,
                z_centering=dataset_opt.z_centering,
                weight_classes=dataset_opt.weight_classes,
            )

            self.test_dataset = CopcInternalDataset(
                os.path.join(dataset_opt.dataroot, 'tiles'),
                split="test",
                transform=self.test_transform,
                train_classes=dataset_opt.train_classes,
                class_map=class_map,
                files=[os.path.join(dataset_opt.dataroot, 'tiles', x) for x in splits["test"]],
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=False,
                do_shift=dataset_opt.do_shift,
                z_centering=dataset_opt.z_centering,
                weight_classes=dataset_opt.weight_classes,
            )
        else:
            files = glob.glob(os.path.join(dataset_opt.dataroot, "**/*.las"), recursive=True)
            files.extend(glob.glob(os.path.join(dataset_opt.dataroot, "**/*.laz"), recursive=True))
            self.test_dataset = CopcInternalDataset(
                dataset_opt.dataroot,
                split="test",
                transform=self.test_transform,
                train_classes=dataset_opt.train_classes,
                files=files,
                is_inference=True,
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=False,
                do_shift=dataset_opt.do_shift,
                z_centering=dataset_opt.z_centering,
                weight_classes=dataset_opt.weight_classes,
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