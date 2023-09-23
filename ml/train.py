'''Train the model'''

import logging
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ml.data_module import Food3DataModule
from ml.lightning_module import Food3LM
from ml.callbacks import (
    CustomProgressBar,
    PrintCallback,
    SamplesVisualizationLogger,
    CustomModelCheckpoint,
)


logger = logging.getLogger(__name__)


def train(cfg) -> None:
    '''Train the model'''

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.trainer.seed)

    food3_data = Food3DataModule(
                        batch_size=cfg.data_module.batch_size,
                        num_workers=cfg.data_module.num_workers,
                    )

    food3_model = Food3LM(
                input_shape=cfg.model.input_shape,
                hidden_units=cfg.model.hidden_units,
                output_shape=cfg.model.output_shape,
            )

    wandb_logger = WandbLogger(
            project=cfg.logger.project,
            name=cfg.logger.name,
            dir=cfg.logger.dir,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        deterministic=cfg.trainer.deterministic,
        logger=wandb_logger,
        callbacks=[
            CustomProgressBar(),
            PrintCallback(),
            SamplesVisualizationLogger(food3_data),
            CustomModelCheckpoint(
                        exp_name=cfg.callbacks.model_checkpoint.exp_name,
                        ckpt_dir=cfg.callbacks.model_checkpoint.ckpt_dir,
                        monitor=cfg.callbacks.model_checkpoint.monitor,
                        mode=cfg.callbacks.model_checkpoint.mode,
                        every=cfg.callbacks.model_checkpoint.every,
                        save_last=cfg.callbacks.model_checkpoint.save_last,
                    ),
        ],
    )

    trainer.fit(food3_model, food3_data)
