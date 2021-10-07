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
from sys import float_info
import re


class CopcInternalDataset(torch.utils.data.Dataset):
    def __init__(self, root, samples, transform, train_classes, resolution, dataset_sampling_rates=[], class_maps={}, is_inference=False, donotcare_class_ids=[], do_augment=False, do_shift=False, augment_transform=None, train_classes_weights=None):

        self.root = root
        self.samples = samples

        # # TODO: Compute nb samples per dataset and total number of samples
        # [dataset,split,bound]
        # for dataset in samples.keys():

        # Compute total number of samples
        self.nb_samples = 0
        for _, splits in samples.items():
            for _, keys in splits.items():
                self.nb_samples += len(keys)

        self.class_maps = class_maps
        self.is_inference = is_inference
        self.resolution = resolution
        self.transform = transform
        self.train_classes = train_classes
        if len(dataset_sampling_rates) > 0 and (len(dataset_sampling_rates) != len(samples.keys())):
            raise Exception("Number of sampling rates must match number of datasets.")
        self.dataset_sampling_rates = dataset_sampling_rates/np.sum(dataset_sampling_rates) # probably of drawing samples from each dataset, sum to 1
        self.donotcare_class_ids = donotcare_class_ids
        self.hunits = 1.0
        self.vunits = 1.0
        self.do_augment = do_augment
        self.do_shift = do_shift
        self.augment_transform = augment_transform
        self.train_classes_weights = torch.Tensor(train_classes_weights)

    # def _map_idx_to_sample(self, idx):

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):

        # Draw a data sample, either randomly or with dataset sample rates
        if not self.is_inference & len(self.dataset_sampling_rates) > 0:
            dataset = np.random.choice(list(self.samples.keys()), p=self.dataset_sampling_rates)
        else:
            dataset = np.random.choice(self.samples.keys(), len(self.samples.keys()))

        split = np.random.choice(list(self.samples[dataset].keys()))
        sample = np.random.choice(list(self.samples[dataset][split].keys()))


        reader = copc.FileReader(os.path.join(self.root,dataset,"copc",split,"octree.copc.laz"))
        header = reader.GetLasHeader()

        # Extract nearest depth, x, and y from sample
        nearest_depth, x, y = re.findall(r'\b\d+\b', sample)
        nearest_depth, x, y = int(nearest_depth), int(x), int(y)
        max_depth = reader.GetDepthAtResolution(self.resolution)
        # Fill z as 0, since we don't care about that dimension
        sample_bounds = copc.Box(copc.VoxelKey(nearest_depth,x,y,0), header)
        # Make the tile 2D
        sample_bounds.z_min = float_info.min
        sample_bounds.z_max = float_info.max

        # If we're doing inference we need to track where the points came from
        if self.is_inference:
            pass
            # TODO [Leo]: REDO this for new splits
            # all_points = []  # Nx3
            # all_points_key = []  # N
            # all_points_idx = []  # N
            # # for depth in range(max_depth):
            # nodes = reader.GetNodesIntersectBox(sample_bounds, resolution=self.resolution)
            # if len(nodes) != max_depth+1:
            #     raise Exception("More nodes than depth levels.")
            # for node in nodes:
            #     points = reader.GetPoints(node)
            #     node_points_within_bounds_idx = []
            #     node_points_within_bounds = []
            #     for id, point in enumerate(points):
            #         if point.Within(sample[1]):
            #             node_points_within_bounds_idx.append(id)
            #             node_points_within_bounds.append(point)
            #
            #     all_points.append(node_points_within_bounds)
            #     all_points_key.append([node.key] * len(node_points_within_bounds))
            #     all_points_idx.append(node_points_within_bounds_idx)
            # # Stack X,Y,Z
            # points = np.stack([[point.X for sublist in all_points for point in sublist], [point.Y for sublist in all_points for point in sublist], [point.Z for sublist in all_points for point in sublist]], axis=1)
            # points_key = np.asarray([node_idx for sublist in all_points_key for node_idx in sublist])
            # points_idx = np.asarray([point_idx for sublist in all_points_idx for point_idx in sublist])
            # classification = np.asarray([point.Classification for sublist in all_points for point in sublist])

        # If training we can just grab points without tracking
        else:
            # Test possible Zs to see if key exist self.samples[dataset][split][sample]
            valid_nodes = []
            for z in self.samples[dataset][split][sample]:
                key = copc.VoxelKey(nearest_depth,x,y,z)
                node = reader.FindNode(key)

                while not node.IsValid() and key.IsValid():
                    key = key.GetParent()
                    node = reader.FindNode(key)
                if node.IsValid():
                    valid_nodes.append(node)

            # Check whether points exist within the x/y region
            # points_xy = reader.GetPointsWithinBox(sample_bounds)

            # Process keys that exist
            copc_points = copc.Points(header)
            loaded_keys = [] # Makes sure we don't load the same key twice
            for node in valid_nodes:
                if node.key.d == nearest_depth:
                    # Get all children points (these will automatically fit
                    copc_points.AddPoints(reader.GetAllChildrenPoints(node.key, self.resolution))
                key = node.key
                while key.IsValid() and key not in loaded_keys:
                    copc_points.AddPoints(reader.GetPoints(key).GetWithin(sample_bounds))
                    loaded_keys.append(key)
                    key = key.GetParent()

            points = np.stack([copc_points.X, copc_points.Y, copc_points.Z], axis=1)
            points_key = []
            points_idx = []
            classification = np.array(copc_points.Classification).astype(np.int)

        y = torch.from_numpy(classification)
        if len(self.class_maps[dataset]):
            y = self._remap_labels(y, self.class_maps[dataset])

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

        data = Data(pos=torch.from_numpy(points).type(torch.float), y=y)

        if self.is_inference:
            data = SaveOriginalPosId()(data)

        if self.do_augment:
            data = self.augment_transform(data)

        if self.transform:
            data = self.transform(data)

        if self.do_shift:
            data = ShiftVoxels()(data)

        return data

    def _remap_labels(self, labels, class_map):
        NUM_CLASSES = 100 # arbitrary
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = labels.clone()

        # first map using the class_map
        for idx in range(NUM_CLASSES):
            if idx not in class_map:
                class_map[idx] = 0

        # add identity mappings for each train class
        for c in self.train_classes:
            class_map[c] = c

        # now shift the existing classes so that they are consecutive
        for i, c in enumerate(self.train_classes):
            for c1, c2 in class_map.items():
                # the existing mapping list maps from one class to another class
                # so we take that second class and check if we want to keep it, i.e. it's in train_classes
                # and if so, we set it to 1+i because i is 0-based and 0 is set to the catch-all class, so we start indexing at 1
                if c2 == c:
                    class_map[c1] = i + 1

        for idx in self.donotcare_class_ids:
            class_map[idx] = IGNORE_LABEL

        for source, target in class_map.items():
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

        if not dataset_opt.is_inference:
            train_samples = {}
            val_samples = {}
            test_samples = {}
            class_maps = {}
            dataset_sampling_rates = []
            for dataset in dataset_opt.datasets:
                dataset_sampling_rates.append(dataset.sampling_rate)
                class_maps[dataset.dataset] = dict(dataset.classification_map.class_map_from_to)
                with open(os.path.join(dataset_opt.dataroot, dataset.dataset, "copc/splits-v%d.json" % (dataset_opt.dataset_version))) as fp:
                    splits = json.load(fp)

                # Training samples
                train_samples[dataset.dataset] = splits["train"]
                val_samples[dataset.dataset] = splits["val"]
                test_samples[dataset.dataset] = splits["test"]

            self.train_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                samples=train_samples,
                transform=self.train_transform,
                train_classes=dataset_opt.training_classes,
                resolution=dataset_opt.resolution,
                dataset_sampling_rates= dataset_sampling_rates,
                class_maps = class_maps,
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=dataset_opt.do_augment,
                do_shift=dataset_opt.do_shift,
                # augment_transform=instantiate_transforms(dataset_opt.augment),
                train_classes_weights=dataset_opt.training_classes_weights,)

            self.val_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                samples=val_samples,
                transform=self.val_transform,
                train_classes=dataset_opt.training_classes,
                resolution=dataset_opt.resolution,
                dataset_sampling_rates=dataset_sampling_rates,
                class_maps=class_maps,
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=False,
                do_shift=dataset_opt.do_shift,
                train_classes_weights=dataset_opt.training_classes_weights,
            )

            self.test_dataset = CopcInternalDataset(
                root=dataset_opt.dataroot,
                samples=test_samples,
                transform=self.test_transform,
                train_classes=dataset_opt.training_classes,
                resolution=dataset_opt.resolution,
                dataset_sampling_rates=dataset_sampling_rates,
                class_maps=class_maps,
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=False,
                do_shift=dataset_opt.do_shift,
                train_classes_weights=dataset_opt.training_classes_weights,
            )
        else:

            self.test_dataset = CopcInternalDataset(
                dataset_opt.dataroot,
                samples= {}, #TODO
                transform=self.test_transform,
                train_classes=dataset_opt.train_classes,
                resolution=dataset_opt.resolution,
                is_inference=True,
                donotcare_class_ids=dataset_opt.donotcare_class_ids,
                do_augment=False,
                do_shift=dataset_opt.do_shift,
                train_classes_weights=dataset_opt.training_classes_weights,
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