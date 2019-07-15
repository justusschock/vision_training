from deliravision.models.gans import AuxiliaryClassifierGANPyTorch
import os
import torch
from training.gans._basic import train, predict

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"real_imgs": "data", "real_labels": "label"}

    model, weight_path = train(AuxiliaryClassifierGANPyTorch,
                               {"latent_dim": 100,
                                "img_size": 28,
                                "n_channels": 1,
                                "n_classes": 10},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               additional_losses={
                                   "auxiliary": torch.nn.CrossEntropyLoss()},
                               key_mapping=key_mapping)

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
