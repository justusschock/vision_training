from deliravision.models.gans import AdversarialAutoEncoderPyTorch
import os
from training.gans._basic import train, predict
import torch

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1500
    key_mapping = {"x": "data"}

    model, weight_path = train(AdversarialAutoEncoderPyTorch,
                               {"latent_dim": 100,
                                "img_shape": (1, 28, 28)},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               additional_losses={
                                   "pixelwise": torch.nn.L1Loss()
                               },
                               key_mapping=key_mapping)

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs,
            generative_network="generator.decoder")
