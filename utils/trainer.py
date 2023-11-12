import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        device: Optional[torch.device] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        log_dir: str = None,
        save_epochs: int = 0,
        regularization: str = None,
        regularization_alpha: float = 0.01,
        amp: bool = False,
        ):
        
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.regularization = regularization
        self.regularization_alpha = regularization_alpha
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device=self.device)

        # auto mixed precision
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()
            
        if log_dir:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.save_epochs = save_epochs

        # self.last_epoch = 1
        # self.file_pattern = None
        # self.milestones = []
        # self.intervals = []
        
    def _compute_regularization(self, alpha=0.01):
        if self.regularization == 'L1':
            return alpha * sum(torch.norm(param, 1) for param in self.model.parameters())
        elif self.regularization == 'L2':
            return alpha * sum(torch.norm(param, 2) for param in self.model.parameters())
        elif self.regularization == 'L2sqrt':
            return alpha * sum(torch.norm(param, 2) ** 2 for param in self.model.parameters())
        else:
            return 0.0

    def evaluate(self, dataloader: DataLoader, epoch: int):
        self.model.eval()

        total_loss = 0.0
        targets = []
        predictions = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                # 计入总损失
                total_loss += loss.item()
                # 记录预测结果
                predictions.extend(y_pred.argmax(dim=1).cpu().numpy())
                # 记录真实结果
                targets.extend(y.argmax(dim=1).cpu().numpy())

        # 计算准确率
        correct_predictions = sum(torch.Tensor(predictions) == torch.Tensor(targets))
        accuracy = correct_predictions.item() / len(targets)
        
        # 将验证损失和准确度写入 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Validation/Loss', total_loss / len(dataloader), epoch)
            self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)

        return total_loss / len(dataloader), accuracy

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()

        total_loss = 0.0
        targets = []
        predictions = []

        for x, y in tqdm(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            if self.amp:
                with autocast():
                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y)

                loss += self._compute_regularization(self.regularization_alpha)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss += self._compute_regularization(self.regularization_alpha)
                loss.backward()
                self.optimizer.step()

            # 计入总损失
            total_loss += loss.item()
            # 记录预测结果
            predictions.extend(y_pred.argmax(dim=1).cpu().numpy())
            # 记录真实结果
            targets.extend(y.argmax(dim=1).cpu().numpy())

        # 计算准确率
        correct_predictions = sum(torch.Tensor(predictions) == torch.Tensor(targets))
        accuracy = correct_predictions.item() / len(targets)

        # 将训练损失和准确度写入 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Train/Loss', total_loss / len(dataloader), epoch)
            self.writer.add_scalar('Train/Accuracy', accuracy, epoch)

        return total_loss / len(dataloader), accuracy

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(self.train_dataloader, epoch)

            if self.val_dataloader is not None:
                val_loss, val_accuracy = self.evaluate(self.val_dataloader, epoch)
                print(f"Epoch {epoch + 1}/{epochs} => "
                    f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} => "
                    f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            if self.writer is not None:
                self.writer.flush()

            if self.lr_scheduler:
                self.lr_scheduler.step()
                
            if self.save_epochs > 0 and (epoch + 1) % self.save_epochs == 0:
                self.save_checkpoint(epoch + 1)

        # self.model_saver.step(avg_loss, val_loss, val_accuracy)

        # if early_stopper is not None:
        #     if early_stopper.early_stop(val_loss):
        #         break

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, f'./ckpts/{epoch}.pth')

    # def set_file_pattern(self, file_pattern):
    #     '''
    #     sets the file pattern for saving the self.model. The file pattern should contain a single {epoch} placeholder.
    #     '''
    #     self.file_pattern = file_pattern
    #     return self

    # def load(self, file):
    #     '''
    #     laods the self.model from the given file.
    #     '''
    #     state = torch.load(file)
    #     self.set_epoch(state['epoch'])
    #     self.model.load_state_dict(state['state_dict'])
    #     return self

    # def load_checkpoint(self):
    #     '''
    #     loads the self.model from the last saved epoch.
    #     '''
    #     file = self.file_pattern.format(epoch=self.last_epoch)
    #     if os.path.isfile(file):
    #         state = torch.load(file)
    #         self.set_epoch(state['epoch'])
    #         self.model.load_state_dict(state['state_dict'])
    #         return True
    #     return False

    # def set_auto_save(self, milestones, intervals):
    #     self.milestones = milestones if milestones is not None else []
    #     self.intervals = intervals if intervals is not None else []
    #     return self
    
    # def save_torchscript(self, file, trace_input_shape=None):
    #     if self.model is None:
    #         raise Exception('self.model is not set')
    #     device = next(self.model.parameters()).device
    #     if trace_input_shape is not None:
    #         mod = torch.jit.trace(self.model, torch.randn(trace_input_shape).to(device))
    #     else:
    #         mod = torch.jit.script(self.model)
    #     mod.save(file)

    # def step(self, loss=0.0, val_loss=0.0, val_accuracy=0.0):
    #     '''
    #     step should be called after each epoch.
    #     '''
    #     self.last_epoch += 1
    #     if self._should_save(self.last_epoch):
    #         state = {
    #             'state_dict': self.model.state_dict(),
    #             'epoch': self.last_epoch,
    #             'loss': loss,
    #             'val_loss': val_loss,
    #             'val_accuracy': val_accuracy
    #         }
    #         file = self.file_pattern.format(epoch=self.last_epoch)

    #         if not os.path.isdir(os.path.dirname(file)):
    #             os.makedirs(os.path.dirname(file))
    #         print(f'saving self.model to: {file}')
    #         torch.save(state, self.file_pattern.format(epoch=self.last_epoch))

    # def count_parameters(self):
    #     '''
    #     returns the number of trainable parameters in the self.model.
    #     '''
    #     # return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # def _should_save(self, epoch):
    #     '''
    #     returns True if the self.model should be saved at the given epoch.
    #     '''
    #     if len(self.intervals) == 0:
    #         return False

    #     for milestone, interval in zip(self.milestones + [float('inf')], self.intervals):
    #         if epoch < milestone:
    #             return epoch % interval == 0

    # def forward(self, x):
    #     return self.model(x)

    # def __getattr__(self, name):
    #     '''
    #     forwards all other attributes to the self.model.
    #     '''
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self.model, name)
