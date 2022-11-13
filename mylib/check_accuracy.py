#!/usr/bin/env python
# coding: utf-8
import mylib.make_dataloader as make_dataloader
import mylib.model_implementation as model_implementation
import torch
from tqdm import tqdm

test_loader = make_dataloader.make_test_loader()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model_implementation.cifar10_alexnet()

model_path = 'data/model.pth'
model.load_state_dict(torch.load(model_path))

model.eval()
test_acc = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_test_acc = test_acc / len(test_loader.dataset)

print(f"the accracy of test dataset is {avg_test_acc * 100}%")