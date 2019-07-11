from deliravision.models.gans import BoundarySeekingGAN
from deliravision.losses import BoundarySeekingLoss
import os
from training.gans._basic import train, predict

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"x": "data"}

    model, weight_path = train(BoundarySeekingGAN, {"latent_dim": 100,
                                                    "img_shape": (1, 28, 28)},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"boundary_seeking":
                                                  BoundarySeekingLoss()})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)