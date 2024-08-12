# Applied mutation op ChangeBatchSize with args: dict_items([('batch_size', 4)])

import json
import sys
import time
from typing import Tuple, Any, Literal, Optional, Dict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

def batched(images: torch.tensor, labels: torch.tensor, batch_size: int):
    for i in range(0, len(labels), batch_size):
        end = min(len(labels), i + batch_size)
        yield (images[i:end], labels[i:end])

def calc_validation(net: nn.Module, data: Dict[str, torch.tensor], criterion, device: str):
    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        for (img, label) in batched(data['img_valid'], data['lbl_valid'], batch_size=64):
            label = label.squeeze()
            with torch.autocast(device_type=device):
                logit = net(img)
                loss = criterion(logit, label.squeeze(-1))
            correct += (torch.max(logit, dim=1)[1] == label).sum().item()
            total_loss += loss.item()
    return (total_loss / len(data['img_valid']), correct / len(data['img_valid']))

def train(net: nn.Module, data: Dict[str, torch.tensor], out_file: Path, num_epochs: int=20, pretrain: Optional[Path]=None):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    if pretrain:
        print('Loading pretrained weights', pretrain)
        w = torch.load(pretrain)
        for (name, param) in w.items():
            if name not in ['fc2.weight', 'fc2.bias', 'fc1.weight', 'fc1.bias']:
                net.state_dict()[name].copy_(param)
    dataset_train = TensorDataset(data['img_train'], data['lbl_train'])
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
    optimizer = optim.Adadelta(net.parameters(), lr=1.0)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    history = []
    val_history = []
    total = len(data['img_train'])
    for epoch in range(0, num_epochs):
        net.train()
        total_loss = 0.0
        correct = 0
        for (img, label) in loader_train:
            label = label.squeeze()
            optimizer.zero_grad()
            with torch.autocast(device_type=device):
                logit = net(img)
                loss = criterion(logit, label.squeeze(-1))
            correct += (torch.max(logit, dim=1)[1] == label).sum().item()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        (val_loss, val_acc) = calc_validation(net, data, criterion, device)
        val_history.append((val_loss, val_acc))
        scheduler.step(np.around(val_loss, 4))
        history.append((total_loss / total, correct / total))
        print(f'Epoch: [{epoch:04}]', f'Loss: {total_loss / total:15.10f}', f'Accuracy: {correct / total:07.3%} ({correct: 6}/{total})', ' | ', f'V_Loss: {val_loss:13.10f}', f'V_Accuracy: {val_acc:07.3%}')
    end = time.time()
    torch.save(net.state_dict(), out_file)
    out_dir = out_file.parent
    torch.save({'epoch': epoch, 'net_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, out_dir / 'checkpoint.tar')
    net_script = torch.jit.script(net)
    net_script.save(out_dir / 'model_script.pth')
    with (out_dir / 'training.json').open('w') as fp:
        json.dump({'duration': end - start, 'final_train_acc': history[-1][1], 'final_valid_acc': val_history[-1][1], 'history': history, 'val_history': val_history}, fp, indent=2)