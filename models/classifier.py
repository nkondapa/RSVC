import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from models.utils import construct_model, modify_model_output_layer
import matplotlib.pyplot as plt


class ClassificationModel(pl.LightningModule):
    def __init__(self, model_type, num_classes, model_kwargs=None, optim_kwargs=None, dataset_params=None):
        super(ClassificationModel, self).__init__()
        self.save_hyperparameters()

        # Instantiate the model
        model = construct_model(model_type, model_kwargs)
        model = modify_model_output_layer(model, num_classes)
        self.model = model
        self.model_kwargs = model_kwargs
        self.optim_kwargs = optim_kwargs
        self.dataset_params = dataset_params  # passed to track transforms applied during training

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr = self.optim_kwargs['lr']
        num_epochs = self.optim_kwargs['num_epochs']
        steps_per_epoch = self.optim_kwargs['steps_per_epoch']
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs,
                                                           steps_per_epoch=steps_per_epoch)
        lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        # fig, axes = plt.subplots(1, 1)
        # axes.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
        # plt.show()
        outputs = self(inputs)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        loss = nn.functional.cross_entropy(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        k = min(5, outputs.shape[1])
        # topk_acc = self.topk_accuracy(outputs, targets, topk=list(range(2, k)))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        # self.log_dict(topk_acc)

    def topk_accuracy(self, output, target, topk=(1,), split='val'):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            # Get the top k predictions from the model output
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            # Check if the true labels are among the top k predictions
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = {}
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                accuracy = correct_k.mul_(100.0 / batch_size)
                res[f'{split}_top{k}_acc'] = accuracy.item()

            return res
