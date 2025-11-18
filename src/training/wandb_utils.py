import wandb
import os

def init_wandb(config: dict, project_name: str = "nlp-mini-project"):
    """
    Initialize a W&B run. 
    - config: dictionary with hyperparameters or settings
    - project_name: wandb project name
    """
    wandb.init(
        project=project_name,
        config=config,
        settings=wandb.Settings(code_dir=os.getcwd())
    )
    return wandb.config


def log_metrics(metrics: dict, step: int = None):
    """
    Log metrics to W&B.
    - metrics: dictionary {"train_loss": 0.5, "val_acc": 0.82}
    - step: optional training step/epoch
    """
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def log_example_audio(audio_array, sample_rate, name="audio_example"):
    """
    Log an audio sample to W&B.
    This is optional, only for audio projects.
    """
    wandb.log({name: wandb.Audio(audio_array, sample_rate=sample_rate)})


def log_example_image(image, name="image_example"):
    """
    Log a spectrogram or image to W&B.
    """
    wandb.log({name: wandb.Image(image)})
