'''Food3LM lightning module'''
# pylint: skip-file

import wandb
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from ml.models.tiny_vgg import TinyVGG


class Food3LM(pl.LightningModule):
    '''Creates the Food3 lightning module.'''

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        '''Initializes the module.'''
        super().__init__()

        self.save_hyperparameters()

        self.model = TinyVGG(input_shape, hidden_units, output_shape)

        self.train_acc = Accuracy(task='multiclass', num_classes=3)
        self.test_acc = Accuracy(task='multiclass', num_classes=3)

        self.loss_fn = nn.CrossEntropyLoss()

        self.log_outputs = {}

    def forward(self, x) -> torch.Tensor:
        '''Forward pass of the model.'''
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''Configures the optimizer.'''
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_index) -> dict:
        '''Training step.'''
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, y)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs) -> None:
        '''Training epoch end.'''
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log_dict({"loss": train_loss, "accuracy": train_acc})
        self.log_outputs["loss"] = train_loss
        self.log_outputs["accuracy"] = train_acc

    def validation_step(self, batch, batch_idx) -> dict:
        '''Validation step.'''
        x, y = batch
        outputs = self(x)
        preds = torch.argmax(outputs, dim=1)

        val_loss = self.loss_fn(outputs, y)
        val_acc = self.test_acc(preds, y)

        return {'val_loss': val_loss, 'val_acc': val_acc,
                'y': y, 'preds': outputs}

    def validation_epoch_end(self, outputs) -> None:
        '''Validation epoch end.'''
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log_dict({"val_loss": val_loss, "val_accuracy": val_acc})
        self.log_outputs["val_loss"] = val_loss
        self.log_outputs["val_accuracy"] = val_acc

        y = torch.cat([x['y'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=preds.cpu().detach().numpy(), y_true=y.cpu().detach().numpy()
                )
            }
        )
