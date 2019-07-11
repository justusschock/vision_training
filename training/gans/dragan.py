from deliravision.models.gans import DRAGAN
from deliravision.losses import GradientPenalty
import os
from training.gans._basic import train, predict

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"x": "data"}

    model, weight_path = train(DRAGAN, {"latent_dim": 100,
                                        "num_channels": 1,
                                        "img_size": 28},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"gradient_penalty":
                                                      GradientPenalty()})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
