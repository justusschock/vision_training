from deliravision.models.gans import LeastSquareGAN
from deliravision.losses import AdversarialLoss
import os
from training.gans._basic import train, predict
import torch

if __name__ == '__main__':

    img_path = os.path.abspath("~/data/")
    outpath = os.path.abspath("~/GanExperiments")
    num_epochs = 1000
    key_mapping = {"imgs": "data"}

    model, weight_path = train(LeastSquareGAN, {"latent_dim": 100,
                                                "img_shape": (1, 28, 28)},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={
                                   "adversarial":
                                       AdversarialLoss(torch.nn.MSELoss())})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)

