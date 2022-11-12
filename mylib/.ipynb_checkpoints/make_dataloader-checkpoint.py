#!/usr/bin/env python
# coding: utf-8
import torchvision
import torchvision.transforms as transforms
import torch

def load_train_data():
    dataset = torchvision.datasets.CIFAR10(
        root = "./data/",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    return dataset

def load_test_data():
    dataset = torchvision.datasets.CIFAR10(
        root = "./data/",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    return dataset

def make_train_loader():
    train_data = load_train_data()
    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = 64,
        shuffle=True,
        num_workers=1
    )
    return train_loader

def make_test_loader():
    test_data = load_test_data()
    test_loader = torch.utils.data.DataLoader(
        dataset = test_data,
        batch_size = 64,
        shuffle=True,
        num_workers=1
    )
    return test_loader

