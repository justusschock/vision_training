from delira.data_loading.dataset import TorchvisionClassificationDataset
from delira.training import PyTorchExperiment, Parameters
import torch
from delira.data_loading import BaseDataManager, RandomSampler, SequentialSampler
from multiprocessing import Process
from torchvision.utils import save_image
import os
from tqdm import tqdm
from deliravision.losses import AdversarialLoss


def create_optims(model, optim_cls,
                  **optim_params):
    return {"generator": optim_cls(model.generator.parameters(),
                                   **optim_params),
            "discriminator": optim_cls(model.discriminator.parameters(),
                                       **optim_params)}


def setup_data(img_path, batchsize, num_processes, transforms_train,
               transforms_val, dset_type="mnist"):
    print("Setting Up Data")
    data = {}
    for train, key, trafo, sampler_cls in zip([True, False], ["train", "val"],
                                              [transforms_train,
                                               transforms_val],
                                              [RandomSampler,
                                               SequentialSampler]):
        dset = TorchvisionClassificationDataset(
            dset_type, img_path,
            train=train, download=True, img_shape=(28, 28))

        dmgr = BaseDataManager(dset, batchsize, num_processes, trafo,
                               sampler_cls=sampler_cls)

        data[key] = dmgr

    return data


def save_data(batches, savepath):
    for idx, batch in enumerate(batches):
        save_image(batch, os.path.join(savepath,
                                       "img_batch_%03d.png" % idx))


def train(model_cls, model_kwargs: dict, outpath: str, data_path, exp_name=None,
          batchsize=64, num_epochs=1500, checkpoint_freq=10,
          additional_losses: dict = None, dset_type="mnist", key_mapping=None,
          create_optim_fn=None):

    if exp_name is None:
        exp_name = model_cls.__name__

    if additional_losses is None:
        additional_losses = {}

    if create_optim_fn is None:
        create_optim_fn = create_optims
    losses = {"adversarial": AdversarialLoss()}
    losses.update(additional_losses)
    params = Parameters(fixed_params={
        "model": {
            **model_kwargs
        },
        "training": {
            "num_epochs": num_epochs,
            "batchsize": batchsize,
            "losses": losses,
            "val_metrics": {},
            "optimizer_cls": torch.optim.Adam,
            "optimizer_params": {"lr": 0.001, "betas": (0.5, 0.9995)},
            "scheduler_cls": None,
            "scheduler_params": {}
        }
    })
    data = setup_data(data_path, params.nested_get("batchsize"), 4, None, None,
                      dset_type)
    exp = PyTorchExperiment(params, model_cls,
                            params.nested_get("num_epochs"),
                            name=exp_name, save_path=outpath,
                            key_mapping=key_mapping,
                            optim_builder=create_optim_fn,
                            checkpoint_freq=checkpoint_freq,
                            gpu_ids=[0])

    model = exp.run(data["train"], data["val"])
    weight_dir = os.path.join(exp.save_path, "checkpoints", "run_00")

    return model, weight_dir


def predict(model, weight_dir, outpath: str, num_epochs: int, exp_name=None,
            checkpoint_freq=10, batchsize=64, num_batches=100, latent_dim=100,
            generative_network="generator"):
    pbar = tqdm(range(0, num_epochs, checkpoint_freq))
    processes = []
    for epoch in pbar:
        weight_file = os.path.join(weight_dir, "checkpoint_epoch_%d.pt" % epoch)
        if not os.path.isfile(weight_file):
            continue
        else:
            model.load_state_dict(torch.load(weight_file)["model"])
        img_save_path = os.path.join(outpath, exp_name,
                                     "epoch_%d" % epoch)
        os.makedirs(img_save_path, exist_ok=True)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        batches = []

        gen_model = model

        if generative_network:
            for model_part in generative_network.split("."):
                gen_model = getattr(gen_model, model_part)

        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                generative_network = model
                preds = gen_model(torch.randn(batchsize, latent_dim,
                                              device=device,
                                              dtype=dtype))
                batches.append(preds.to("cpu"))

        proc = Process(target=save_data, args=(batches, img_save_path))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for _proc in processes:
        _proc.join()
