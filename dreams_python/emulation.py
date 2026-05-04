'''
Normalizing Flows functionality

This package is essentially a wrapper around the NeHOD package.
We recommend you download that prior to beginning work with the emulators
https://github.com/trivnguyen/nehod_torch

Written by Alex M. Garcia (alexgarcia@virginia.edu) with thanks to Jonah Rose
for providing some functions that have been adapted here. And Tri Nguyen for
developing the NeHOD emulator
'''
## standard imports
import os
import sys
import numpy as np
import pandas as pd
from glob import glob

## torch
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from torch.utils.data import DataLoader, TensorDataset

## optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

## Zuko and NeHOD
import zuko
from nehod_torch import datasets ## https://github.com/trivnguyen/nehod_torch
from nehod_torch.nehod import train_utils

class NPE(pl.LightningModule):
    def __init__(
        self, 
        context_dim, in_dim,
        num_transforms=8,
        projection_dims=[64, 64],
        hidden_dims=[64, 64, 64, 64],
        dropout=0.05,
        lr=1e-4,
        wd=1e-2
    ):
        super().__init__()
        # Lightning allows extra information to be saved in with the model
        # this is a convenient way to save the normalization statistics
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd

        self.lin_proj_layers = nn.ModuleList()
        for i in range(len(projection_dims)):
            in_proj_dim = context_dim if i == 0 else projection_dims[i - 1]
            out_proj_dim = projection_dims[i]
            self.lin_proj_layers.append(nn.Linear(in_proj_dim, out_proj_dim))
            self.lin_proj_layers.append(nn.ReLU())
            self.lin_proj_layers.append(nn.BatchNorm1d(out_proj_dim))
            self.lin_proj_layers.append(nn.Dropout(dropout))
        self.lin_proj_layers = nn.Sequential(*self.lin_proj_layers)
        self.flow = zuko.flows.NSF(
            in_dim, projection_dims[-1], transforms=num_transforms,
            hidden_features=hidden_dims, randperm=True
        )

    def forward(self, context):
        embed_context = self.lin_proj_layers(context)
        return embed_context

    def training_step(self, batch, batch_idx):
        target, context = batch
        embed_context = self.forward(context)
        loss = -self.flow(embed_context).log_prob(target).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=target.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        target, context = batch
        embed_context = self.forward(context)
        loss = -self.flow(embed_context).log_prob(target).mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=target.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.wd
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50_000, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class emulator():
    def __init__(self, features, labels, emulator_name, _data_root='emulator'):
        self._data_root = _data_root
        if not os.path.exists(self._data_root):
            print(f'Creating Directory {self._data_root}/ ...')
            os.mkdir(self._data_root)

        if not os.path.exists(f'{self._data_root}/flows/'):
            print(f'Creating Directory {self._data_root}/flows/ ...')
            os.mkdir(f'{self._data_root}/flows/')
        
        self.emulator_name = emulator_name
        if not os.path.exists(f'{self._data_root}/flows/{self.emulator_name}'):
            print(f'Creating Directory {self._data_root}/flows/{self.emulator_name} ...')
            os.mkdir(f'{self._data_root}/flows/{self.emulator_name}')
        
        self.features = features
        self.labels = labels

        self.n_features = features.shape[1]
        self.n_labels = labels.shape[1]

        feature_parameters = [f'feature_{i}' for i in range(self.n_features)]
        label_parameters = [f'label_{i}' for i in range(self.n_labels)]
        self.header = ','.join(feature_parameters + label_parameters)

        data_cond = np.hstack([features, labels])
        np.savetxt(f"{self._data_root}/flows/{self.emulator_name}/{self.emulator_name}_cond.csv",
                   data_cond, delimiter=',', header=self.header, comments='')
        
        # since we only care about the central in this emulator, this line is useless but necessary
        np.savez(f"{self._data_root}/flows/{self.emulator_name}/{self.emulator_name}",
                 features=np.ones((len(data_cond),10,5)), mask=np.ones((len(data_cond),10),dtype=bool)) 
        
        _, self.data, _, norm_dict = datasets.read_preprocess_dataset(
            data_root=f"{self._data_root}/flows/{self.emulator_name}/", data_name=self.emulator_name, flag=None,
            conditioning_parameters=label_parameters + feature_parameters
        )

        self.target = self.data[:, :self.n_labels]
        self.cond = self.data[:, self.n_labels:]

        self.new_norm_dict = {
            'target_mean': np.array(norm_dict['cond_mean'][:self.n_labels]),
            'target_std' : np.array(norm_dict['cond_std'][:self.n_labels]),
            'cond_mean'  : np.array(norm_dict['cond_mean'][self.n_labels:]),
            'cond_std'   : np.array(norm_dict['cond_std'][self.n_labels:]),
        }
        
        target_unnorm = self.target.numpy() * self.new_norm_dict['target_std'] + self.new_norm_dict['target_mean']
        cond_unnorm   = self.cond.numpy() * self.new_norm_dict['cond_std'] + self.new_norm_dict['cond_mean']
        
        return

    def train_flow(
        self,
        train_split=0.85,
        validate_split=0.15,
        test_split=0.0,
        seed=1234,
        max_steps=5000,
        accelerator='gpu',
        enable_progress_bar=False,
        inference_mode=False,
        val_check_interval=20,
        check_val_every_n_epoch=None,
        log_every_n_steps=10,
        num_workers=1,
        callbacks=None,
        ckpt_path=None, ## add ckpt_path to resume training
        ### calibrated hyperparameters
        num_transforms=8,
        projection_dim=64,
        hidden_dim=64,
        n_projection_layers=2,
        n_hidden_layers=4,
        dropout=0.05,
        lr=1e-4,
        decay=1e-2,
        batch_size=128
    ):
        assert( (train_split + validate_split + test_split) == 1 )
        
        nsims = len(self.target)
        sim_idx = np.arange(nsims)
        rng = np.random.default_rng(seed)
        rng.shuffle(sim_idx)
        
        target_train = self.target[sim_idx][:int(nsims*train_split)]
        cond_train = self.cond[sim_idx][:int(nsims*train_split)]
        
        target_valid = self.target[sim_idx][int(nsims*train_split):int(nsims*train_split)+int(nsims*validate_split)]
        cond_valid = self.cond[sim_idx][int(nsims*train_split):int(nsims*train_split)+int(nsims*validate_split)]
        
        target_test = self.target[sim_idx][int(nsims*train_split)+int(nsims*validate_split):]
        cond_test = self.cond[sim_idx][int(nsims*train_split)+int(nsims*validate_split):]
        
        data_loader_train = datasets.create_dataloader(
            (target_train, cond_train), batch_size=batch_size,
            shuffle=True, pin_memory=torch.cuda.is_available(),
            num_workers=num_workers)
        
        data_loader_valid = datasets.create_dataloader(
            (target_valid, cond_valid), batch_size=batch_size,
            shuffle=False, pin_memory=torch.cuda.is_available(),
            num_workers=num_workers)
        
        data_loader_test = datasets.create_dataloader(
            (target_test, cond_test), batch_size=batch_size,
            shuffle=False, pin_memory=torch.cuda.is_available(),
            num_workers=num_workers)

        pl.seed_everything(seed)

        model = NPE(
            context_dim=self.n_features,
            in_dim=self.n_labels,
            num_transforms=num_transforms,
            projection_dims=[projection_dim] * n_projection_layers,
            hidden_dims=[hidden_dim] * n_hidden_layers,
            dropout=dropout,
            lr=lr,
            wd=decay
        )

        if callbacks is None:
            callbacks = [ ## default set-up
                pl.callbacks.ModelCheckpoint(
                    filename="{epoch}-{val_loss:.4f}", monitor='val_loss',
                    save_top_k=1, mode='min', save_weights_only=False,
                    save_last=True),
                pl.callbacks.EarlyStopping(
                    monitor='val_loss', patience=500,   # in steps, given val_check_interval=20
                    mode='min', verbose=True),
            ]

        train_logger = pl_loggers.CSVLogger(f'{self._data_root}/flows/{self.emulator_name}/', version='')

        # Define a PyTorch Lightning Trainer, which will handle the entire training process
        trainer = pl.Trainer(
            default_root_dir=f'{self._data_root}/flows/{self.emulator_name}/',
            max_steps=max_steps,
            accelerator=accelerator,
            callbacks=callbacks,
            logger=train_logger,
            enable_progress_bar=enable_progress_bar,
            inference_mode=inference_mode,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=log_every_n_steps,
        )
        
        # train the model
        trainer.fit(
            model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_valid, 
            ckpt_path=ckpt_path
        )
        
        return

    def loss_data(self):
        loss_data = np.genfromtxt(f'{self._data_root}/flows/{self.emulator_name}/lightning_logs/metrics.csv', delimiter=',')

        tcut = np.isnan(loss_data[:,2])
        train_loss = loss_data[~tcut,2]
        train_epoch = loss_data[~tcut,0]
        
        vcut = np.isnan(loss_data[:,4])
        valid_loss = loss_data[~vcut,4]
        valid_epoch = loss_data[~vcut, 0]
        
        return (train_loss, train_epoch), (valid_loss, valid_epoch)

    def load_trained_flow(self, specific_checkpoint=None):
        model_dir = f'{self._data_root}/flows/{self.emulator_name}/lightning_logs/checkpoints/'
        if specific_checkpoint is None:
            ckpt_files = glob(model_dir + "*.ckpt")
            available_models = [f for f in ckpt_files if "last" not in f]
            if len(available_models) > 1:
                print(f'More than 1 checkpoint exists, defaulting to first one: {available_models[0]}')
            elif len(available_models) == 0:
                print(f'No checkpoint exists at {model_dir}')
                return -1
            specific_checkpoint = available_models[0]
        return NPE.load_from_checkpoint(specific_checkpoint)

    def make_prediction(self, samples, n_samples=1, specific_checkpoint=None):
        model  = self.load_trained_flow(specific_checkpoint=specific_checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
    
        context_norm = (samples - self.new_norm_dict['cond_mean']) / self.new_norm_dict['cond_std']
        context_norm = torch.tensor(context_norm, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            flow_context = model(context_norm)
            model_samples = model.flow(flow_context).sample((n_samples,)).cpu().numpy()
            # shape: (n_samples, n_context_points, n_outputs)
    
        data_emulated = model_samples * self.new_norm_dict['target_std'] + self.new_norm_dict['target_mean']
        return data_emulated  # (n_samples, n_context_points, n_outputs)

    def tune_flow(
        self,
        n_trials=50,
        train_split=0.85,
        validate_split=0.15,
        seed=1234,
        max_steps=5000,
        num_workers=1,
        accelerator='gpu',
        # search space bounds
        num_transforms_range=(1, 12),
        projection_dim_range=(32, 128),
        hidden_dim_range=(32, 128),
        n_projection_layers_range=(1, 3),
        n_hidden_layers_range=(2, 8),
        dropout_range=(1e-4, 0.5),
        lr_range=(1e-5, 1e-1),
        decay_range=(1e-4, 0.5),
        batch_range=[32, 64, 96, 128, 256, 512]
    ):
        nsims = len(self.target)
        sim_idx = np.arange(nsims)

        rng = np.random.default_rng(seed)
        rng.shuffle(sim_idx)
    
        # Pre-split data once, outside the objective
        target_train = self.target[sim_idx][:int(nsims*train_split)]
        cond_train   = self.cond[sim_idx][:int(nsims*train_split)]
        target_valid = self.target[sim_idx][int(nsims*train_split):]
        cond_valid   = self.cond[sim_idx][int(nsims*train_split):]
    
        def objective(trial):
            # Sample hyperparameters
            num_transforms   = trial.suggest_int('num_transforms', *num_transforms_range)
            n_proj_layers    = trial.suggest_int('n_projection_layers', *n_projection_layers_range)
            proj_dim         = trial.suggest_int('projection_dim', *projection_dim_range, step=32)
            n_hidden_layers  = trial.suggest_int('n_hidden_layers', *n_hidden_layers_range)
            hidden_dim       = trial.suggest_int('hidden_dim', *hidden_dim_range, step=32)
            dropout          = trial.suggest_float('dropout', *dropout_range)
            lr               = trial.suggest_float('lr', *lr_range, log=True)
            wd               = trial.suggest_float('wd', *decay_range, log=True)
            batch_size = trial.suggest_categorical(
                'batch_size', batch_range
            )
    
            projection_dims = [proj_dim] * n_proj_layers
            hidden_dims     = [hidden_dim] * n_hidden_layers
    
            data_loader_train = datasets.create_dataloader(
                (target_train, cond_train), batch_size=batch_size,
                shuffle=True, pin_memory=torch.cuda.is_available(),
                num_workers=num_workers)
    
            data_loader_valid = datasets.create_dataloader(
                (target_valid, cond_valid), batch_size=batch_size,
                shuffle=False, pin_memory=torch.cuda.is_available(),
                num_workers=num_workers)
    
            pl.seed_everything(seed)
    
            model = NPE(
                context_dim=self.n_features,
                in_dim=self.n_labels,
                num_transforms=num_transforms,
                projection_dims=projection_dims,
                hidden_dims=hidden_dims,
                dropout=dropout,
                lr=lr,
                wd=wd
            )
    
            callbacks = [
                pl.callbacks.EarlyStopping(
                    monitor='val_loss', patience=200,
                    mode='min', verbose=False),
                # PyTorchLightningPruningCallback(trial, monitor='val_loss'),
            ]
    
            trainer = pl.Trainer(
                max_steps=max_steps,
                accelerator=accelerator,
                callbacks=callbacks,
                enable_progress_bar=False,
                inference_mode=False,
                val_check_interval=20,
                check_val_every_n_epoch=None,
                log_every_n_steps=10,
                logger=False,  # suppress logging during tuning
            )
    
            trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_valid)
    
            val_loss = trainer.callback_metrics.get('val_loss')
            if val_loss is None:
                return float('inf')
            return val_loss.item()  # remove trial.report and trial.should_prune

        storage_dir = f'{self._data_root}/flows/{self.emulator_name}/optuna/'
        if not os.path.exists(storage_dir):
            os.mkdir(storage_dir)
        study = optuna.create_study(
            study_name=f"{self.emulator_name}_tuning",
            storage=f"sqlite:///{storage_dir}/optuna.db",
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200),
        )
        study.optimize(objective, n_trials=n_trials)
    
        self.best_params = study.best_params
        np.save(f'{self._data_root}/flows/{self.emulator_name}/best_params.npy', self.best_params, allow_pickle=True)
        return study

if __name__ == "__main__":
    print('Hello World!')