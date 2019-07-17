from deliravision.models.gans import EnergyBasedGAN
from deliravision.losses import PullAwayLoss, DiscriminatorMarginLoss
import os
from training.gans._basic import train, predict
import torch

if __name__ == '__main__':

    img_path = os.path.abspath("~/data/")
    outpath = os.path.abspath("~/GanExperiments")
    num_epochs = 1000
    key_mapping = {"imgs": "data"}

    model, weight_path = train(EnergyBasedGAN, {"latent_dim": 100,
                                                "num_channels": 1,
                                                "img_size": 28},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"pullaway":
                                                      PullAwayLoss(),
                                                  "pixelwise":
                                                      torch.nn.MSELoss(),
                                                  "discriminator_margin":
                                                      DiscriminatorMarginLoss()
                                                  })

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
