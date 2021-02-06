import logging
import os
import time
import copy
import warnings
import torch


class AbcTrainer:
    def __call__(self):
        raise NotImplementedError


class BaseTrainer(AbcTrainer):
    def __init__(self,
                 model,
                 modeltag,
                 criterion=None,
                 metric_ops={},
                 optimizer=None,
                 dataloader=None,
                 validation_dataloader=None,
                 num_epochs=None,
                 scheduler=None,
                 checkpointer=None,
                 checkpoint_dir=None,
                 checkpoint_period=None,
                 device=None,
                 logger=None):
        self.model = model
        self.modeltag = modeltag

        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_ops = metric_ops

        if validation_dataloader:
            self.dataloaders = {'train': dataloader, 'val': validation_dataloader}
            self.phases = ['train', 'val']
            self.best_model_wts = None
            self.best_loss = float('inf')
        else:
            self.dataloaders = {'train': dataloader}
            self.phases = ['train']

        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period
        self.device = device
        
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.modeltag or __name__)

    def __call__(self):
        since = time.time()

        for epoch_i in range(self.num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch_i, self.num_epochs - 1))
            self.logger.info('-' * 50)

            for phase in self.phases:
                epoch_accum_loss = 0.
                epoch_accum_metric_values = dict(zip(self.metric_ops.keys(), [0.] * len(self.metric_ops.keys())))

                for i, data in enumerate(self.dataloaders[phase], 0):
                    inputs, targets = data
                    inputs = inputs.to(self.device)
                    targets = [target.to(self.device) for target in targets]

                    if phase == 'train':
                        self.model.train()
                        self.optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            loss = self.model(inputs, targets)
                            loss.backward()
                            self.optimizer.step()
                    elif phase == 'val':
                        self.model.train()
                        with torch.set_grad_enabled(False):
                            loss = self.model(inputs, targets)
                        self.model.eval()
                        with torch.set_grad_enabled(False):
                            predictions = self.model(inputs)
                            for metric_name, metric_op in self.metric_ops.items():
                                epoch_accum_metric_values[metric_name] += metric_op(predictions, targets)

                    epoch_accum_loss += loss.item()
                        
                epoch_loss = epoch_accum_loss / len(self.dataloaders[phase].dataset)
                self.logger.info('[{}] Loss: {:.4f}'.format(phase, epoch_loss))
                
                if phase == 'val':
                    for metric_name, metric_value in epoch_accum_metric_values.items():
                        epoch_loss = epoch_accum_loss / len(self.dataloaders[phase].dataset)
                        self.logger.info('[{}] {}: {:.4f}'.format(phase, metric_name, metric_value))

                    if epoch_loss < self.best_loss:
                        self.best_loss = epoch_loss
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        
        time_elapsed = time.time() - since
        self.logger.info('Training completed. Time Cost: {:.0f}m {:0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        if 'val' in self.phases:
            self.logger.info('Best Validation Loss: {:.4f}'.format(self.best_loss))

        best_model_wts = self.best_model_wts or self.model.state_dict()
        self._save_checkpoint_for_state_dict(best_model_wts)

    def _save_checkpoint_for_state_dict(self, state_dict):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        if not os.path.isdir(self.checkpoint_dir):
            self.logger.warning('The checkpoint save directory is invalid')
        
        filepath = os.path.join(self.checkpoint_dir,
                                '{}.pth.{}'.format(self.modeltag, int(time.time() * 1000)))
        torch.save(state_dict, filepath)