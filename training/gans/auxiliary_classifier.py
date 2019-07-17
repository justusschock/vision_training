from deliravision.models.gans import AuxiliaryClassifierGANPyTorch
import os
import torch
from training.gans._basic import train, predict

if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"real_imgs": "data", "real_labels": "label"}
    latent_dim = 100
    batchsize = 64
    n_classes = 10

    model, weight_path = train(AuxiliaryClassifierGANPyTorch,
                               {"latent_dim": latent_dim,
                                "img_size": 28,
                                "n_channels": 1,
                                "n_classes": n_classes},
                               os.path.join(outpath, "train"), img_path,
                               num_epochs=num_epochs,
                               additional_losses={
                                   "auxiliary": torch.nn.CrossEntropyLoss()},
                               key_mapping=key_mapping,
                               batchsize=batchsize)

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs,
            gen_fns=[torch.randn, torch.randint],
            gen_args=[(batchsize, latent_dim), (0, n_classes, (batchsize, ))],
            gen_kwargs=[{}, {"dtype": torch.long}])
