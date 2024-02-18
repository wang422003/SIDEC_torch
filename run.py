import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from __init__ import *
from experiment import VAEXperiment, VDEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, VDEDataset, load_edges
from utils import structural_inference_pipeline, calculate_auroc, store_adj_and_results
from arg_parser import parse_args
# from pytorch_lightning.plugins import DDPPlugin


args = parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=args.save_folder_path,
                              name=args.tb_name,)

# For reproducibility
seed_everything(args.seed, True)

# Dataset setup
data = VDEDataset(args=args, pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()

edges = load_edges(args)

model = vae_models['VDE'](
    input_size=args.dims,
    hidden_size=args.hidden,
    dropout_rate=args.dropout,
)

experiment = VDEXperiment(
    vae_model=model,
    edges=edges,
    params=config['exp_params']
)

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy="ddp",
                 accelerator=args.device,
                 max_epochs=args.epochs)
                 # **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)

# get latent
predict = runner.predict(experiment, dataloaders=data.test_dataloader(), return_predictions=True)

# structural inference with ppcor
adjs = structural_inference_pipeline(predict, args)

# calculate the AUROC
auroc_score, ls_auroc_score = calculate_auroc(adjs, edges)

# save results
store_adj_and_results(adjs, edges, ls_auroc_score, args)

