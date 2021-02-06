import time
import copy
import torch

from .base_trainer import BaseTrainer


class DetectorTrainer(BaseTrainer):
    def __call__(self):
        since = time.time()

        for epoch_i in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch_i, self.num_epochs - 1))
            print('-' * 10)

            for phase in self.phases:
                if phase == 'train':
                    self.model.train()
                elif phase == 'val':
                    self.model.eval()
            
                acc_running_loss = .0
                for i, data in enumerate(self.dataloaders[phase], 0):
                    inputs, targets = data
                    inputs = inputs.to(self.device)
                    targets = [target.to(self.device) for target in targets]
                    
                    batch_size = len(inputs)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if self.criterion is None:
                            loss = self.model(inputs, targets)
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    acc_running_loss += loss.item() * batch_size

                epoch_loss = acc_running_loss / len(self.dataloaders[phase].dataset)
                print('[{}] Loss: {:.4f}'.format(phase, epoch_loss))
                
                if phase == 'train':
                    self.scheduler.step()
                elif phase == 'val':
                    if epoch_loss < self.best_loss:
                        self.best_loss = epoch_loss
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        
        time_elapsed = time.time() - since
        print('Training completed. Time Cost:  {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if 'val' in self.phases:
            print('Best Validation Loss: {:.4f}'.format(self.best_loss))
            self.model.load_state_dict(self.best_model_wts)

        if self.checkpoint_path:
            torch.save(self.model.state_dict(), self.checkpoint_path)