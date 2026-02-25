
"""
This contains all of the functions that are critical for processing vision data
It is stored here because we have spaghetti code as it is
It is designed to work with CIFAR predominantly. Bite me
"""

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pickle
import os


def get_coarse_labels(dataset):
    """
    Fetches superclass information from CIFAR100
    """
    file = "train" if dataset.train else "test"
    path = os.path.join(dataset.root, "cifar-100-python", file)

    with open(path, "rb") as f:
        entry = pickle.load(f, encoding = "latin1")

    return entry["coarse_labels"]


def split_dataset_by_tasks(dataset, n_tasks = 20):
    """
    Splits dataset into incremental tasks based on classes

    Args:
        dataset: torchvision dataset
        class_order: optional list specifying class order
        n_tasks: number of tasks

    Returns:
        List of subset datasets, one per task
    """
    task_datasets = []

    coarse_labels = get_coarse_labels(dataset)
    coarse_labels = np.array(coarse_labels)

    for task_id in range(n_tasks):
        indices = np.where(coarse_labels == task_id)[0]
        task_subset = Subset(dataset, indices)
        task_datasets.append(task_subset)

    return task_datasets


def process_subset(subset, task_id):
    """
    Changes the labels on a torchvision dataset

    Args:
        subset: subset item 
        task_id: value to set
    """
    data = torch.stack([x.clone() for x, _ in subset])
    task_labels = torch.full((data.size(0),), task_id, dtype = torch.long)
    return {
        "images": data,
        "labels": task_labels,
    }


def fetch_transforms():
    """ Returns two transform objects """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    return [transform_train, transform_test]


def prepare_gate_datasets(tasks):
    """
    Changes the label on task datasets for use in training with the TEG
    """
    tasks = [process_subset(subset, i) for i, subset in enumerate(tasks)]

    inputs = torch.cat([t["images"] for t in tasks])
    targets = torch.cat([t["labels"] for t in tasks])

    return TensorDataset(inputs, targets)

