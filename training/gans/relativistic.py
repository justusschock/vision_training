from deliravision.models.gans import RelativisticGAN
from deliravision.losses import AdversarialLoss
import os
from training.gans._basic import train, predict
import torch

if __name__ == '__main__':
    img_path = os.path.abspath("~/data/")
    outpath = os.path.abspath("~/GanExperiments")
    num_epochs = 1000
    key_mapping = {"x": "data"}

    model, weight_path = train(RelativisticGAN,
                               {"latent_dim": 100,
                                "img_size": 28,
                                "num_channels": 1},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={
                                   "adversarial": AdversarialLoss(
                                       torch.nn.BCEWithLogitsLoss())})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
