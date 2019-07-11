from deliravision.models.gans import ConditionalGAN
from deliravision.losses import AdversarialLoss
import torch
import os
from training.gans._basic import train, predict

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"x": "data", "labels": "label"}

    model, weight_path = train(ConditionalGAN, {"latent_dim": 100,
                                                "n_classes": 10,
                                                "img_shape": (1, 28, 28)},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               key_mapping=key_mapping,
                               additional_losses={"adversarial":
                                   AdversarialLoss(torch.nn.MSELoss())})

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs)
