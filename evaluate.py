import torch
from utils import AHE
import cv2
import lazy_dataset
from utils import collate
from model import ResNet

from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
db = AHE()

t_ds = db.get_dataset('testing_set')
t_ds = lazy_dataset.new(t_ds)
def prepare_example(example):
    path = example['image_path']
    img = cv2.imread(path,1)
    example['image'] = img
    return example

def prepare_dataset(dataset,batch_size=1):
    dataset = dataset.map(prepare_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(collate)
    return dataset


model = ResNet(n_classes=10).to(device)
PATH = torch.load('ckpt_best_loss.pth')
model.load_state_dict(PATH)
model.eval()
with torch.no_grad():
    t_ds = prepare_dataset(t_ds,batch_size=1)
    correct = 0
    for batch in tqdm(t_ds):
        images = batch['image']
        images = torch.Tensor(images).to(device=device)
        targets = torch.Tensor(batch['target']).to(device=device)
        outputs = model(images)
        if torch.argmax(outputs).item() == targets.item():
                correct += 1
    print(f'testing done...')
    print(f'model accuracy on testing set: {correct * 100 / len(t_ds)}')

