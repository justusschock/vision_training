from deliravision.models.gans import BoundaryEquilibriumGAN
from deliravision.losses import BELoss
import os
from training.gans._basic import train, predict

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"x": "data"}

    model, weight_path = train(BoundaryEquilibriumGAN, {"latent_dim": 100,
                                                        "img_size": 28,
                                                        "n_channels": 1},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"began": BELoss()})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
