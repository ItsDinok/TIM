import numpy as np
import torch
from basic_gate import task_estimation_gate, GatedResNet32
__all__ = ["build_experiment"]

def build_experiment(backend, root = "/.data", device = "cuda", classes_per_task = 3):
    """
    This builds the experimental environment

    arguments:
        - backend: a data backend used for specific datasets
        - root: where the data is stored
        - device: pytorch device
        - classes_per_task: number of classes per task

    returns:
        - dictionary of experiment variables (see dictionary)
    """
    # Set system state
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    trainset, testset = backend.load(root)

    task_bundles, global_map = backend.build_tasks(
        root = root,
        train_transform = trainset.transform,
        test_transform = testset.transform
    )

    # Prepare data for teg
    gate_train_tasks = backend.split_gate(trainset)
    gate_test_tasks = backend.split_gate(testset)
    gate_train = backend.prepare_gate(gate_train_tasks)
    gate_test = backend.prepare_gate(gate_test_tasks)

    task_map = {str(i): classes_per_task for i in range(len(task_bundles))}

    # Construct models
    model = GatedResNet32(task_map).to(device)
    teg = task_estimation_gate(
        tasks = len(gate_train_tasks),
        device = device,
        train_data = gate_train,
        test_data = gate_test
    )

    # Create task sets
    train_tasks = [b.train for b in task_bundles]
    test_tasks = [b.test for b in task_bundles]

    task_label_maps = _build_task_label_maps(train_tasks)

    return {
        "model": model,
        "teg": teg,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "task_label_maps": task_label_maps
    }


def _build_task_label_maps(train_tasks):
    """
    INTERNAL ONLY
    Constructs the task label maps

    arguments:
        - train_tasks: list of tasks
    returns:
        - task_label_maps: a dictionary with tasks as keys and their corresponding labels
    """
    task_label_maps = {}

    for task_id, train_task in enumerate(train_tasks):
        labels = sorted({int(y) for _, y in train_task})
        task_label_maps[task_id] = {g: i for i, g in enumerate(labels)}

    return task_label_maps