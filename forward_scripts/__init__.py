from forward_scripts.forward import CopcInference
from forward_scripts.forward_binary import CopcBinaryInference


def predict_file(
    checkpoint_dir,
    in_file_path,
    out_file_path,
    wandb_run,
    cuda,
    hUnits,
    vUnits,
    upsample=False,
    override_all=False,
    confidence_threshold=0.0,
):
    if isinstance(wandb_run, list) and len(wandb_run) == 1:
        wandb_run = wandb_run[0]

    forward_class = CopcBinaryInference if isinstance(wandb_run, list) else CopcInference

    # the defaults of all these arguments are probably fine for prod (change upsample on/off as desired)
    inference = forward_class(
        checkpoint_dir,
        debug=False,
        upsample=upsample,
        confidence_threshold=confidence_threshold,
    )

    # all these args are required except override_all
    inference.run_inference(
        in_file_path=in_file_path,
        out_file_path=out_file_path,
        cuda=cuda,
        wandb_run=wandb_run,
        hUnits=hUnits,
        vUnits=vUnits,
        override_all=override_all,
    )
