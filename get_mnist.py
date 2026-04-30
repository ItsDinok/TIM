from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms

TASKS: Dict[int, List[int]] = {
    0: [6, 9, 0],
    1: [2, 5, 7],
    2: [3, 8, 4]
}

DROPPED = {1}

class MultiTaskDataset(Dataset):
    """
    Wraps a base dataset for multi-task learning.
    - Returns image
    - Returns original class label
    - Returns task identity
    """

    def __init__(self, task_dataset: Dataset, task_id: int, label_map: Dict[int, int] = None):
        self.task_dataset = task_dataset
        self.task_id = task_id
        self.label_map = label_map

    def __len__(self):
        return len(self.task_dataset)

    def __getitem__(self, idx):
        x, y = self.task_dataset[idx]
        return x, int(y), self.task_id


@dataclass
class TaskDatasetBundle:
    task_id: int
    train: Dataset
    test: Dataset
    classes: List[int]

class RemappedSubset(Dataset):
    """
    A subset wrapper that remaps labels.

    Useful for:
    - Local task labels: [6, 9, 0] -> [1, 2, 3]
    - global contiguous labels: all classes -> [0 ... 8]
    """
    def __init__(self, base_dataset: Dataset, indices: List[int], label_map: Dict[int, int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map
        self.dataset = self.base_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[self.indices[idx]]
        return x, int(y)


class FilteredTaskDataset(Dataset):
    """
    Dataset for one task.
    Keeps only the task's digit classes and remaps them to local labels
    Example:
        task classes [6, 9. 0] -> labels {6:0, 9:1, 0:2}
    """
    def __init__(self, base_dataset: Dataset, classes: Sequence[int]):
        self.base_dataset = base_dataset
        self.classes = list(classes)
        self.class_to_local = {c: i for i, c in enumerate(classes)}

        targets = self._get_targets(base_dataset)
        mask = torch.zeros(len(targets), dtype = torch.bool)
        for c in self.classes:
            mask |= (targets == c)

        self.indices = torch.where(mask)[0].tolist()


    def _get_targets(self, dataset: Dataset) -> torch.Tensor:
        if hasattr(dataset, "targets"):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets
            return torch.tensor(targets)
        raise AttributeError("Dataset does not expose a .targets attribute.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        y = self.class_to_local[int(y)]
        return x, y


def fetch_transforms():
    """
    These have to adapt the training data to work with a ResNet, which expects three channels
    I know a ResNet is overkill, but this is the very lowest test I can think of
    """
    transform_train = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels = 3),
        transforms.RandomCrop(28, padding = 2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])

    transform_test = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels = 3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])

    return transform_train, transform_test


class GateTaskDataset(Dataset):
    """
    Wraps a class dataset but replaces class label with task label
    Used to train TEG modules
    """
    def __init__(self, task_dataset: Dataset, task_id: int):
        self.task_dataset = task_dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.task_dataset)

    def __getitem__(self, idx):
        x = self.task_dataset[idx][0]
        return x, self.task_id


def _get_targets(dataset: Dataset) -> torch.Tensor:
    """
    Extract targets from torchvision MNIST dataset as a tensor
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            return torch.tensor(targets)
        return targets.clone() if isinstance(targets, torch.Tensor) else torch.tensor(targets)

    raise AttributeError("Dataset does not expose a .targets attribute.")


def _make_global_label_map(tasks: Dict[int, List[int]]) -> Dict[int, int]:
    """
    Build continuous global label map across all kept classes.

    Example:
        digits used = [0, 2, 3, 4, 5, 6, 7, 8, 9]
        map -> {0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:0}
    """
    all_classes = sorted({c for cls_list in tasks.values() for c in cls_list})
    return {original_class: new_label for new_label, original_class in enumerate(all_classes)}


def _filter_indices_by_classes(dataset: Dataset, allowed_classes: List[int]) -> List[int]:
    """
    Return dataset indices whose labels belong to allowed classes.
    """
    targets = _get_targets(dataset)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in allowed_classes:
        mask |= (targets == c)
    return torch.where(mask)[0].tolist()


def build_mnist_cil_tasks(
        root: str = "./data",
        train_transform = None,
        test_transform = None,
        download: bool = True,
) -> Tuple[List[TaskDatasetBundle], Dict[int, int]]:
    """
    Build MNIST class-incrmeental learning tasks
    Args:
        root: dataset root directory
        train_transform: transform for train split
        test_transform: transform for test split
        download: whether to download MNIST if needed

    Returns:
        task_bundles: list of TaskDatasetBundle
        global_label_map: mapping from original digit -> contiguous global labels
    """
    if train_transform is None:
        train_transform = transforms.ToTensor()
    if test_transform is None:
        test_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root = root,
        train = True,
        download = download,
        transform = train_transform
    )

    test_dataset = datasets.MNIST(
        root = root,
        train = False,
        download = download,
        transform = test_transform
    )

    global_label_map = _make_global_label_map(tasks=TASKS)
    task_bundles: List[TaskDatasetBundle] = []

    for task_id, classes in TASKS.items():
        train_indices = _filter_indices_by_classes(train_dataset, classes)
        test_indices = _filter_indices_by_classes(test_dataset, classes)

        task_train = MultiTaskDataset(
            Subset(train_dataset, train_indices),
            task_id = task_id
        )

        task_test = MultiTaskDataset(
            Subset(test_dataset, test_indices),
            task_id = task_id
        )

        task_bundles.append(
            TaskDatasetBundle(
                task_id = task_id,
                train = task_train,
                test = task_test,
                classes = classes
            )
        )

    return task_bundles, global_label_map


def split_dataset_by_tasks(dataset: Dataset, task_definitions: Dict[int, List[int]] = TASKS):
    """
    Returns a list of datasets, one per task
    Each task dataset uses local labels [0, 1, 2]
    """
    task_datasets = []
    for task_id in sorted(task_definitions.keys()):
        classes = task_definitions[task_id]
        task_datasets.append(FilteredTaskDataset(dataset, classes))
    return task_datasets


def prepare_gate_datasets(task_datasets: Sequence[Dataset]):
    """
    Converts per-task class datasets into one dataset for gate training
    Where labels are task identities instead of class identities.
    """
    gate_sets = []
    for task_id, ds in enumerate(task_datasets):
        gate_sets.append(GateTaskDataset(ds, task_id))
    return ConcatDataset(gate_sets)


def load_mnist(root: str = "./data"):
    transform_train, transform_test = fetch_transforms()

    trainset = datasets.MNIST(
        root = root,
        train = True,
        download = True,
        transform = transform_train
    )

    testset = datasets.MNIST(
        root = root,
        train = False,
        download = True,
        transform = transform_test
    )

    return trainset, testset


def get_task_class_sequence() -> List[List[int]]:
    """
    Returns the original digit classes per task
    """
    return [TASKS[i] for i in sorted(TASKS.keys())]


def get_all_kept_classes() -> List[int]:
    """
    Returns all classes used across tasks, excluding dropped ones
    """
    return sorted({c for cls in TASKS.values() for c in cls if c not in DROPPED})


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding = 2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tasks, global_map = build_mnist_cil_tasks(
        test_transform = test_transform,
        train_transform = train_transform,
        root = "./data"
    )

    print("Dropped classes:", DROPPED)
    print("Task definition:")
    for task in tasks:
        print(
            f"Task: {task.task_id}: classes = {task.classes}, "
            f"train_size = {len(task.train)}, test_size = {len(task.test)}"
        )

    print("Global label map:", global_map)
