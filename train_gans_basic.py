from delira.data_loading.dataset import TorchvisionClassificationDataset
from deliravision.models.gans import DeepConvolutionalGAN

from delira.training import PyTorchExperiment, Parameters
import torch
from delira.data_loading import BaseDataManager, RandomSampler, SequentialSampler


def create_optims(model: DeepConvolutionalGAN, optim_cls,
                  **optim_params):
    return {"generator": optim_cls(model.generator.parameters(),
                                   **optim_params),
            "discriminator": optim_cls(model.discriminator.parameters(),
                                       **optim_params)}


class AdversarialLoss(torch.nn.Module):
    def __init__(self, loss_fn=torch.nn.BCELoss()):
        super().__init__()
        self._loss_fn = loss_fn

    def forward(self, pred, gt):

        gt = torch.ones_like(pred) * int(gt)

        return self._loss_fn(pred, gt)


def setup_data(batchsize, num_processes, transforms_train, transforms_val):
    print("Setting Up Data")
    data = {}
    for train, key, trafo, sampler_cls in zip([True, False], ["train", "val"],
                                              [transforms_train,
                                               transforms_val],
                                              [RandomSampler,
                                               SequentialSampler]):
        dset = TorchvisionClassificationDataset(
            "mnist", "C:\\Users\\JSC7RNG\\Downloads\\data",
            train=train, download=True, img_shape=(64, 64))

        dmgr = BaseDataManager(dset, batchsize, num_processes, trafo,
                               sampler_cls=sampler_cls)

        data[key] = dmgr

    return data


if __name__ == '__main__':
    from torchvision.utils import save_image
    import os
    from tqdm import tqdm

    outpath = r"./gan_experiments"
    weight_dir = ""
    exp_name = "DCGAN"
    num_batches = 500
    batchsize = 64
    latent_dim = 100
    num_epochs = 500
    checkpoint_freq = 10

    if not weight_dir:
        params = Parameters(fixed_params={
            "model": {
                "latent_dim": latent_dim,
                "num_channels": 1,
                "img_size": 64
            },
            "training": {
                "num_epochs": num_epochs,
                "batchsize": batchsize,
                "losses": {"adversarial": AdversarialLoss()},
                "val_metrics": {},
                "optimizer_cls": torch.optim.Adam,
                "optimizer_params": {"lr": 0.0002, "betas": (0., 0.9995)},
                "scheduler_cls": None,
                "scheduler_params": {}
            }
        })
        data = setup_data(params.nested_get("batchsize"), 4, None, None)
        exp = PyTorchExperiment(params, DeepConvolutionalGAN,
                                params.nested_get("num_epochs"),
                                name=exp_name, save_path=outpath,
                                key_mapping={"imgs": "data"},
                                optim_builder=create_optims,
                                checkpoint_freq=checkpoint_freq,
                                gpu_ids=[0])

        model = exp.run(data["train"], data["val"])
        weight_dir = os.path.join(exp.save_path, "checkpoints", "run_00")

    else:
        model = DeepConvolutionalGAN(latent_dim, 64, 1)

        model.load_state_dict(torch.load(weights)["model"])

    for epoch in tqdm(range(0, num_epochs, checkpoint_freq)):
        weight_file = os.path.join(weight_dir, "checkpoint_epoch_%d.pt" % epoch)
        img_save_path = os.path.join(outpath, "preds", exp_name,
                                     "epoch_%d" % epoch)
        os.makedirs(img_save_path, exist_ok=True)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        for i in range(num_batches):
            preds = model.generator(torch.rand(batchsize, latent_dim,
                                               device=device, dtype=dtype))

            save_image(preds, os.path.join(img_save_path,
                                           "img_batch_%03d.png" % i))
