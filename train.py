import os
import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler


def evalute(model, criterion, dataloader):
    # 测试模型
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向过程开启 autocast
            with autocast():
                logits = model(inputs)
                loss += criterion(logits, targets).item()

            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    # loss = loss / len(dataloader)
    return loss, accuracy

def train_epoch(model, criterion, optimizer, dataloader, **kwargs):
    model.train()
    grad_scaler = kwargs['grad_scaler'] if 'grad_scaler' in kwargs else None
    l2_reg = kwargs['l2_reg'] if 'l2_reg' in kwargs else None

    # iteratable = tqdm(enumerate(dataloader)) if with_tqdm else enumerate(dataloader)
    device = next(model.parameters()).device
    epoch_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向过程开启 autocast
        with autocast():
            logits = model(inputs)
            loss = criterion(logits, targets)

        # 添加 L2 正则化
        if l2_reg is not None:
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.norm(param)
            loss += l2_reg * l2_loss

        # 反向传播在 autocast 上下文之外
        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

# 训练模型
def train(model_saver, criterion, lr_scheduler, start_epoch, end_epoch, train_loader, test_loader, **kwargs):
    # print(f'Training...')

    model = model_saver.model
    optimizer = lr_scheduler.optimizer
    writer = kwargs['writer'] if 'writer' in kwargs else None
    early_stopper = kwargs['early_stopper'] if 'early_stopper' in kwargs else None

    for epoch in range(start_epoch, end_epoch):
        epoch_loss = train_epoch(model, criterion, optimizer, train_loader)
        avg_loss = epoch_loss / len(train_loader)    
        val_loss, val_accuracy = evalute(model, criterion, test_loader)
        model_saver.step(avg_loss, val_loss, val_accuracy)

        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('LR', lr_scheduler.get_last_lr()[0], epoch)
            writer.flush()

        print(f'Epoch [{epoch+1}/{end_epoch}] LR:{lr_scheduler.get_last_lr()} Loss/train:{epoch_loss:.4f} Loss/val:{val_loss:.4f} Accuracy/val:{val_accuracy:.2f}%')

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                break

        # noise.step()
        lr_scheduler.step()
