import os
import torch
import torch.nn as nn

class ModelProxy(nn.Module):   
    '''
    ModelSaver is a utility class for saving models at specified intervals.
    '''
    def __init__(self, model=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.last_epoch = 1
        self.file_pattern = None
        self.milestones = []
        self.intervals = []

    def wrap(self, model):
        self.model = model
        return self
    
    def set_epoch(self, epoch):
        self.last_epoch = epoch if epoch is not None and epoch > 0 else 1
        return self

    def set_file_pattern(self, file_pattern):
        '''
        sets the file pattern for saving the model. The file pattern should contain a single {epoch} placeholder.
        '''
        self.file_pattern = file_pattern
        return self

    def load(self, file):
        '''
        loads the model from the given epoch.
        '''
        state = torch.load(file)
        self.set_epoch(state['epoch'])
        self.model.load_state_dict(state['state_dict'])
        return self
        
    def set_auto_save(self, milestones, intervals):
        self.milestones = milestones if milestones is not None else []
        self.intervals = intervals if intervals is not None else []
        return self

    def step(self, loss=0.0, val_loss=0.0, val_accuracy=0.0):
        '''
        step should be called after each epoch.
        '''
        self.last_epoch += 1
        if self._should_save(self.last_epoch):
            state = {
                'state_dict': self.model.state_dict(),
                'epoch': self.last_epoch,
                'loss': loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            file = self.file_pattern.format(epoch=self.last_epoch)

            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            print(f'saving model to: {file}')
            torch.save(state, self.file_pattern.format(epoch=self.last_epoch))

    def count_parameters(self):
        '''
        returns the number of trainable parameters in the model.
        '''
        # return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _should_save(self, epoch):
        '''
        returns True if the model should be saved at the given epoch.
        '''
        if len(self.intervals) == 0:
            return False

        for milestone, interval in zip(self.milestones + [float('inf')], self.intervals):
            if epoch < milestone:
                return epoch % interval == 0

    def forward(self, x):
        return self.model(x)

    def __getattr__(self, name):
        '''
        forwards all other attributes to the model.
        '''
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
