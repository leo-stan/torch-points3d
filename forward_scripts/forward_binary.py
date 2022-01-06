from forward_scripts.forward import CopcInference, get_args
import torch
import os
import wandb
from omegaconf import OmegaConf, DictConfig

from torch_points3d.metrics.model_checkpoint import ModelCheckpoint


class CopcBinaryInference(CopcInference):
    def _do_model_inference(self, models, device, data):
        """Runs a batch of data through multiple models

        Args:
            models (list[BaseModel]): list of models to run inference on
            device (str): cpu/cuda
            data (pyg.Batch): data to run inference on

        Returns:
            list[Tensor]: list of pre-softmax model outputs for each model
        """
        outs = []
        for model in models:
            data_clone = data.clone()
            model.set_input(data_clone, device)
            model.forward()
            out = model.get_output()
            outs.append(out)
        return outs

    def _extract_model_predictions(self, reverse_class_maps, data, outputs):
        """From a list of pre-softmax model outputs, aggregate the results
        using some metric to come up with final predictions
        """
        # outputs is a list of model outputs of size cxN, where c varies for each output
        # the length of outputs is the number of models that were run
        model_probs_list = [torch.nn.functional.softmax(output, dim=1) for output in outputs]
        # 2d array with N rows and len(outputs) columns,
        # whose value is the class with the maximum probability for that model and point
        model_probs_max_class = torch.cat(
            [torch.max(model_prob, dim=1)[1].unsqueeze(1) for model_prob in model_probs_list], dim=1
        )
        # the probability for each class above ^
        model_probs_max_class_probs = torch.cat(
            [torch.max(model_prob, dim=1)[0].unsqueeze(1) for model_prob in model_probs_list], dim=1
        )
        # binary mask of all the points who got classified as "other" by all the models
        other_class_mask = model_probs_max_class[:] == 0
        other_class_row_mask = torch.all(other_class_mask, dim=1)

        num_points = len(other_class_row_mask)
        # our final predictions
        preds = torch.zeros(num_points, device=model_probs_max_class.device).long()

        # set the probability of choosing other class to 0, since we're only interested in non-other classes
        model_probs_max_class_probs[other_class_mask] = 0
        # model_probs_max_class_probs gives us a matrix with the maximum probability from each model
        # and model_probs_max_class gives us the class with the maximum probability within each model
        # now, we need to get the model with the maximum probaility
        # i.e. if we have two models that both identify positively, we want to use the "most confident" model
        # therefore, max_class_model_idx gives us a list whos values are the index of the model that is most confident
        max_model_probs, max_class_model_idx = torch.max(model_probs_max_class_probs, dim=1)
        # we also need to get the list of actual class values that we selected, so we index model_probs_max_class
        # to get the class of the most confident model
        max_class = model_probs_max_class[torch.arange(len(model_probs_max_class)), max_class_model_idx]

        # now, we apply the inverse class mapping for each model to get our final model results
        for model_id in range(len(outputs)):
            # we're only interested in the current model
            model_mask = max_class_model_idx == model_id
            # get all the predicted labels from the current model
            # (and add one since the reverse_class_map assumes that the '0' class is 'ignore')
            orig_labels = max_class[model_mask] + 1
            new_labels = orig_labels.clone()

            # apply the inverse class map
            for source, target in reverse_class_maps[model_id]:
                mask = orig_labels == source
                new_labels[mask] = target

            # update our prediction with our true labels
            preds[model_mask] = new_labels

        # on all the rows where all models predicted "other" class, then set their predictions to be class 1
        preds[other_class_row_mask] = 1
        # we also want to set the positive predictions that fall below the confidence threshold to unclassified
        preds[max_model_probs <= self.confidence_threshold] = 0

        return self._upsample_preds_and_convert_to_dict(data, preds.cpu().numpy())

    def _init_model_from_wandb(self, in_file_path, out_file_path, wandb_runs, hUnits, vUnits, override_all, device):
        """Initializes multiple models from a list of wandb paths"""
        if not isinstance(wandb_runs, list):
            raise ValueError("ForwardBinary wandb_runs should be a list!")

        print("Downloading multiple wandb run weights...")
        data_config = None
        models = []
        reverse_class_maps = []
        # download each run, load the checkpoint and create a run
        for wandb_run in wandb_runs:
            run_checkpoint_dir = os.path.join(self.checkpoint_dir, wandb_run)
            wandb.restore(self.model_name + ".pt", run_path=wandb_run, root=run_checkpoint_dir)

            checkpoint = ModelCheckpoint(run_checkpoint_dir, self.model_name, self.metric, strict=True)

            if data_config is None:
                data_config = OmegaConf.to_container(checkpoint.data_config, resolve=True)
                data_config["is_inference"] = True

            model = checkpoint.create_model(DictConfig(checkpoint.dataset_properties), weight_name=self.metric)
            model.eval()
            model = model.to(device)
            models.append(model)

            # provide a default class map in case the config doesn't have one
            class_map = checkpoint.data_config.reverse_class_map
            if not class_map:
                if "/building-v1/" in wandb_run:
                    class_map = [[2, 6]]
                elif "/veg-binary/" in wandb_run:
                    class_map = [[2, 5]]

            reverse_class_maps.append(class_map)

        dataset = self._init_dset(in_file_path, hUnits, vUnits, data_config, models[0])
        print(dataset)

        self._run_inference(
            models,
            dataset,
            device,
            in_file_path,
            out_file_path,
            reverse_class_maps,
            override_all,
        )


if __name__ == "__main__":

    args = get_args()

    # the defaults of all these arguments are probably fine for prod (change upsample on/off as desired)
    inference = CopcBinaryInference(
        args.checkpoint_dir,
        model_name=args.model_name,
        metric=args.metric,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        debug=args.debug,
        upsample=args.upsample,
        confidence_threshold=args.confidence_threshold,
    )

    # all these args are required except override_all
    inference.run_inference(
        in_file_path=args.in_file_path,
        out_file_path=args.out_file_path,
        cuda=args.cuda,
        wandb_run=[
            "rock-robotic/building-v1/15e45z0x",
            "rock-robotic/COPC-ground-v1/31kc98wj",
            "rock-robotic/veg-binary/7jot397i",
        ],
        hUnits=args.hUnits,
        vUnits=args.vUnits,
        override_all=args.override_all,
    )
