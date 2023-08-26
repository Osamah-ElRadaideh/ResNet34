import numpy as np
import torch
import torch.nn as nn
from model import ResNet
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from utils import AHE, collate
import lazy_dataset
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sw = SummaryWriter()
#load the files
#####CIFAR loader


db = AHE()
train_ds = db.get_dataset('training_set')
val_ds = db.get_dataset('validation_set')

def load_img(example):
    img = cv2.imread(example['image_path'])
    example['image'] = img.astype(np.float32)
    return example

def prepare_dataset(dataset,batch_size=16):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(collate)
    return dataset

train_ds = prepare_dataset(train_ds)
val_ds = prepare_dataset(val_ds,batch_size=1)
#model hyperparamters
# 100 classes for CIFAR
#4 classes for MCCD
#10 classes for AHE
model = ResNet(n_classes=10).to(device)
optim = torch.optim.Adam(model.parameters(),lr=1e-3)
CE = nn.CrossEntropyLoss()
# train_loader = DataLoader(train_ds, batch_size=32, drop_last=True, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=1)
path = 'ckpt_best_loss.pth'
val = 1e7
#one epoch step
def train_step(train_loader):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optim.zero_grad()
        # x, y = batch
        x = torch.tensor(np.array(batch['image']))
        y = torch.tensor(batch['target'])
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = CE(output, y)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optim.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def val_step(val_loader):
    model.eval()
    epoch_loss = 0
    accuracy = 0
    with torch.inference_mode():
        for batch in val_loader:
            # x, y = batch
            x = torch.tensor(np.array(batch['image']))
            y = torch.tensor(batch['target'])
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = CE(output, y)
            epoch_loss += loss.item()
            if torch.argmax(output).item() == y:
                accuracy +=1
    return epoch_loss / len(val_loader), accuracy * 100/ len(val_loader)

for i in tqdm(range(250)):
    train_loss = train_step(train_ds)
    validation_loss, validation_accuracy = val_step(val_ds)
    if validation_accuracy < val:
        val = validation_accuracy
        torch.save(model.state_dict(), path)
    print (f'epoch {i+1} training and validation losses: {train_loss} |--|" {validation_loss} |--| {validation_accuracy} ')
    sw.add_scalar("training/loss", train_loss, i)
    sw.add_scalar('validation/loss', validation_loss, i)
    sw.add_scalar('validation/accuracy', validation_accuracy, i)