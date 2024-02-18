import os
import math
import torch
import numpy as np
from torch import optim
from models import BaseVAE, VDE
from types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, labels = batch
        self.curr_device = t_0.device

        results = self.forward(t_0, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, labels = batch
        self.curr_device = t_0.device

        results = self.forward(t_0, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  # t_0.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, labels = batch
        self.curr_device = t_0.device

        results = self.forward(t_0, labels=labels)
        test_loss = self.model.loss_function(*results,
                                             M_N=1.0,  # t_0.shape[0]/ self.num_test_imgs,
                                             optimizer_idx=optimizer_idx,
                                             batch_idx=batch_idx)

        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


class VDEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: VDE,
                 edges: np.ndarray,
                 params: dict) -> None:
        super(VDEXperiment, self).__init__()

        self.model = vae_model
        self.edges = edges
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, t_1 = batch
        self.curr_device = t_0.device

        results = self.forward(t_0)
        # results = results.append(t_1)
        train_loss = self.model.loss_function(results,
                                              t_1,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, t_1 = batch
        self.curr_device = t_0.device

        results = self.forward(t_0)
        # print("Results t0 from forward pass: ", type(results))
        # print(len(results))
        # print(results[0].size())
        # print(results[1].size())
        # print(results[2].size())
        # print(results[3].size())

        # results = results.append(t_1)

        # print("Results t0 + t1 from forward pass: ", type(results))
        # print(len(results))
        val_loss = self.model.loss_function(results,
                                            t_1,
                                            M_N=1.0,  # t_0.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        pass

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, t_1 = batch
        self.curr_device = t_0.device

        results = self.forward(t_0)
        # results = results.append(t_1)
        test_loss = self.model.loss_function(results,
                                             t_1,
                                             M_N=1.0,  # t_0.shape[0]/ self.num_test_imgs,
                                             optimizer_idx=optimizer_idx,
                                             batch_idx=batch_idx)

        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

    def predict_step(self, batch, batch_idx, optimizer_idx=0):
        t_0, t_1 = batch
        self.curr_device = t_0.device
        # print("t_0: ", t_0.size())

        results = self.model.check_latent(t_0)
        return results
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


if __name__ == "__main__":
    print("This is the experiment.py file.")