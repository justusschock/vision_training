from deliravision.models.gans import WassersteinGradientPenaltyGAN
from deliravision.losses import GradientPenalty
import os
from training.gans._basic import train, predict
import torch

if __name__ == '__main__':
    img_path = os.path.abspath("~/data/")
    outpath = os.path.abspath("~/GanExperiments")
    num_epochs = 1000
    key_mapping = {"x": "data"}
    torch.autograd.set_detect_anomaly(True)

    model, weight_path = train(WassersteinGradientPenaltyGAN,
                               {"latent_dim": 100,
                                "img_shape": (1, 28, 28)},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"gradient_penalty":
                                                  GradientPenalty()})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
