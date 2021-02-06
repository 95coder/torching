class BaseTrainer:
    def __init__(self,
                 model,
                 optimizer=None, 
                 train_dataloader=None, 
                 criterion=None,
                 validation_dataloader=None,
                 num_epochs=10, 
                 scheduler=None,
                 checkpoint_path=None,
                 device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        if validation_dataloader:
            self.dataloaders = {'train': train_dataloader, 'val': validation_dataloader}
            self.phases = ['train', 'val']
            self.best_model_wts = None
            self.best_loss = float('inf')
        else:
            self.dataloaders = {'train': train_dataloader}
            self.phases = ['train']

        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.device = device

    def __call__(self):
        raise NotImplementedError
