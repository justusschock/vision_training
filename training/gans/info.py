from deliravision.models.gans import InfoGAN
from deliravision.losses import AdversarialLoss
import os
from training.gans._basic import train, predict
from itertools import chain as iterchain
from deliravision.utils.tensor_ops import make_onehot_torch
import torch


def create_optims(model, optim_cls, **optim_params):
    return {"generator": optim_cls(model.generator.parameters(),
                                   **optim_params),
            "discriminator": optim_cls(model.discriminator.parameters(),
                                       **optim_params),
            "info": optim_cls(iterchain(model.generator.parameters(),
                                        model.discriminator.parameters()),
                              **optim_params)}


def uniform(shape, device, dtype, min_val, max_val):
    _code = torch.empty(*shape,
                        device=device, dtype=dtype)
    _code.uniform_(min_val, max_val)
    return _code


def onehot_ints(max_val, shape, device, dtype):
    _labels = torch.randint(max_val,
                            shape,
                            device=device,
                            dtype=dtype)

    if _labels.size(-1) != max_val:
        _labels = make_onehot_torch(_labels, n_classes=max_val)

    return _labels


if __name__ == '__main__':

    img_path = "~/data/"
    outpath = "~/GanExperiments"
    num_epochs = 1000
    key_mapping = {"imgs": "data"}
    batchsize = 64
    latent_dim = 100
    n_classes = 10
    code_dim = 2

    model, weight_path = train(InfoGAN, {"latent_dim": latent_dim,
                                         "num_channels": 1,
                                         "img_size": 28,
                                         "n_classes": n_classes,
                                         "code_dim": code_dim},
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
                               create_optim_fn=create_optims,
                               batchsize=batchsize)

    predict(model, weight_path, os.path.join(outpath, "preds"), num_epochs,
            gen_fns=[torch.randn, onehot_ints, uniform],
            gen_args=[(batchsize, latent_dim),
                      (n_classes, (batchsize, 1)),
                      ((batchsize, code_dim),)],
            gen_kwargs=[{}, {"dtype": torch.long}, {}])
