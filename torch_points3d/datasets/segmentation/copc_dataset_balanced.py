from torch_points3d.datasets.segmentation.copc_dataset import CopcInternalDataset,CopcBaseDatasetFactory


class CopcInternalDatasetBalanced(CopcInternalDataset):
    def compute_sample_probabilities(self, samples):
        self.sample_probability = {}
        for dset_name, dataset in self.datasets.items():
            # zero-out the sample probabilities
            self.sample_probability[dset_name] = [0] * len(samples[dset_name])
            for i, sample in enumerate(samples[dset_name]):
                label_hist = {label: 0 for label in self.dataset_options.training_classes}
                label_hist[0] = 0

                # Remap the labels in the histogram so we have a common comparison across datasets
                for dataset_label, count in sample.label_hist.items():
                    label_hist[self.remapping_dict[dset_name][int(dataset_label)]] += count
                # This code is very much prototype, for now if the number of point for a particular label is >100
                # we multiply the number of samples of that label by an arbitrary ratio. The idea being, the more
                # sample of a wanted class the higher the probability of drawing that sample.
                if 6 in label_hist.keys() and label_hist[6] > 100:
                    self.sample_probability[dset_name][i] += self.dataset_options.training_class_ratios[3] * label_hist[6]
                elif 3 in label_hist.keys() and label_hist[3] > 100:
                    self.sample_probability[dset_name][i] += self.dataset_options.training_class_ratios[2] * label_hist[3]
                elif 2 in label_hist.keys() and label_hist[2] > 100:
                    self.sample_probability[dset_name][i] += self.dataset_options.training_class_ratios[1] * label_hist[2]
                else:
                    self.sample_probability[dset_name][i] += self.dataset_options.training_class_ratios[0] * label_hist[0]
        # flatten sample probabilities across datasets
        self.sample_probability = [item for sublist in self.sample_probability.values() for item in sublist]


class CopcDatasetBalancedFactory(CopcBaseDatasetFactory):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt, CopcInternalDatasetBalanced)
