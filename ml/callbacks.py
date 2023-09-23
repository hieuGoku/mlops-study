'''Custom callbacks for PyTorch Lightning'''

import os
import wandb
import torch
from pytorch_lightning.callbacks import TQDMProgressBar, Callback
from ml.utils import check_exist_dir


class CustomProgressBar(TQDMProgressBar):
    '''Custom progress bar for PyTorch Lightning.'''
    def on_train_epoch_start(self, trainer, pl_module):
        '''Training epoch start.'''
        self.main_progress_bar.set_description(
            f'Epoch {self.trainer.current_epoch + 1}')


class PrintCallback(Callback):
    '''Print callback for PyTorch Lightning.'''

    def on_train_start(self, trainer, pl_module) -> None:
        '''Training start.'''
        pl_module.print('------------------- Training start -------------------')
        return super().on_train_start(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        '''Training epoch start.'''
        pl_module.print(f'Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}')
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        '''Training epoch end.'''
        train_loss = pl_module.log_outputs["loss"]
        train_acc = pl_module.log_outputs["accuracy"]
        val_loss = pl_module.log_outputs["val_loss"]
        val_acc = pl_module.log_outputs["val_accuracy"]
        pl_module.print(f"loss: {train_loss:.3f} - accuracy: {train_acc:.3f} - val_loss: {val_loss:.3f} - val_accuracy: {val_acc:.3f}")
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        '''Training end.'''
        pl_module.print('------------------- Training end -------------------')
        return super().on_train_end(trainer, pl_module)


class SamplesVisualizationLogger(Callback):
    '''Samples visualisation logger for PyTorch Lightning.'''

    def __init__(self, datamodule) -> None:
        '''Initializes the callback.'''
        super().__init__()
        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        '''Validation end.'''
        # Get validation batch
        val_batch = next(iter(self.datamodule.val_dataloader()))
        x, y = val_batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)

        # Get the predictions
        outputs = pl_module(x)
        preds = torch.argmax(outputs, dim=1)

        # log images and predictions as a W&B Table
        columns = ['image', 'label', 'pred']
        data = [[wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(x, y, preds))]

        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(columns=columns, data=data),
                "epoch": trainer.current_epoch,
            }
        )


class CustomModelCheckpoint(Callback):
    '''Model checkpoint callback for PyTorch Lightning.'''

    def __init__(
            self,
            exp_name,
            ckpt_dir: str = "./checkpoints",
            monitor: str = "val_loss",
            mode: str = "min",
            every: int = 1,
            save_last: bool = True,
        ) -> None:
        '''Initializes the callback.'''

        super().__init__()
        self.exp_name = exp_name
        self.exp_dir = ckpt_dir + f"/{self.exp_name}"
        check_exist_dir(self.exp_dir)
        self.monitor = monitor
        self.mode = mode
        self.every = every
        self.save_last = save_last
        self.best_score = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        '''Validation end.'''
        score = trainer.callback_metrics[self.monitor]

        if self.best_score is None:
            self.best_score = score
            self.save_best_checkpoint(trainer, pl_module)
        elif self.mode == "min" and score < self.best_score:
            self.best_score = score
            self.save_best_checkpoint(trainer, pl_module)
        elif self.mode == "max" and score > self.best_score:
            self.best_score = score
            self.save_best_checkpoint(trainer, pl_module)

        self.save_every_n_epochs(trainer, pl_module, self.every)

        if self.save_last:
            trainer.save_checkpoint(
                    self.exp_dir + "/last.ckpt"
                )

        if trainer.current_epoch + 1 == trainer.max_epochs:
            self.save_model(pl_module, f"ml/weights/pt/{self.exp_name}.pt")

    def save_every_n_epochs(self, trainer, pl_module, every) -> None:
        '''Saves a checkpoint every n epochs.'''
        if (trainer.current_epoch + 1) % every == 0:
            trainer.save_checkpoint(
                self.exp_dir + f"/epoch_{trainer.current_epoch + 1}.ckpt"
            )

    def save_best_checkpoint(self, trainer, pl_module) -> None:
        '''Saves a checkpoint.'''
        epoch = trainer.current_epoch + 1
        trainer.save_checkpoint(f'{self.exp_dir}/best-{self.monitor}.ckpt')
        pl_module.print(
            f"Saved best checkpoint at epoch {epoch} with {self.monitor} {self.best_score:.3f}"
        )

    def save_model(self, model, save_dir: str) -> None:
        '''Saves a model.'''
        torch.save(model.state_dict(), save_dir)
        print(f"Saved model to {save_dir}")
