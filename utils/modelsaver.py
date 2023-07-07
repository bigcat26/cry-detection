import os
import torch

class ModelSaver(object):
    '''
    ModelSaver is a utility class for saving models at specified intervals.
    '''
    def __init__(self, model, path_pattern, auto_load, last_epoch=1, intervals=None, milestones=None):
        '''
        model: model to save
        path_pattern: pattern for the path to save the model to. Should contain a single {epoch} placeholder.
        auto_load: if True, loads the model from the last saved checkpoint.
        last_epoch: last epoch at which the model was saved. If None, defaults to 1.
        milestones: list of epochs at which to save the model. If None, defaults to [].
        intervals: list of epochs between saves. Should be the same length as milestones. If None, defaults to [10].
        '''
        self.model = model
        self.path_pattern = path_pattern
        self.milestones = milestones if milestones is not None else []
        self.intervals = intervals if intervals is not None else [10]
        self.last_epoch = last_epoch

        if auto_load:
            file = self.path_pattern.format(epoch=last_epoch)
            if os.path.isfile(file):
                print(f'loading model from: {file}')
                state = torch.load(file)
                self.model.load_state_dict(state['state_dict'])

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
            file = self.path_pattern.format(epoch=self.last_epoch)
            
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            print(f'saving model to: {file}')
            torch.save(state, self.path_pattern.format(epoch=self.last_epoch))

    def _should_save(self, epoch):
        '''
        returns True if the model should be saved at the given epoch.
        '''
        for milestone, interval in zip(self.milestones + [float('inf')], self.intervals):
            if epoch < milestone:
                return epoch % interval == 0
