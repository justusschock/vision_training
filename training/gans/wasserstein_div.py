from deliravision.models.gans import WassersteinDivergenceGAN
from deliravision.losses import WassersteinDivergence
import os
from training.gans._basic import train, predict
import torch

if __name__ == '__main__':
    img_path = os.path.abspath("~/data/")
    outpath = os.path.abspath("~/GanExperiments")
    num_epochs = 1000
    key_mapping = {"x": "data"}
    torch.autograd.set_detect_anomaly(True)

    model, weight_path = train(WassersteinDivergenceGAN,
                               {"latent_dim": 100,
                                "img_shape": (1, 28, 28)},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"divergence":
                                                  WassersteinDivergence()})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
