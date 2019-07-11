from deliravision.models.gans import InfoGAN
from deliravision.losses import AdversarialLoss
import os
from training.gans._basic import train, predict
import torch
from itertools import chain as iterchain


def create_optims(model, optim_cls, **optim_params):
    return {"generator": optim_cls(model.generator.parameters(),
                                   **optim_params),
            "discriminator": optim_cls(model.discriminator.parameters(),
                                       **optim_params),
            "info": optim_cls(iterchain(model.generator.parameters(),
                              model.discriminator.parameters()),
                              **optim_params)}


if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"imgs": "data"}

    model, weight_path = train(InfoGAN, {"latent_dim": 100,
                                         "num_channels": 1,
                                         "img_size": 28,
                                         "n_classes": 10,
                                         "code_dim": 2},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={
                                   "categorical":
                                       torch.nn.CrossEntropyLoss(),
                                   "continuous":
                                       torch.nn.MSELoss(),
                                   "adversarial":
                                       AdversarialLoss(torch.nn.MSELoss())
                               },
                               create_optim_fn=create_optims)

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
