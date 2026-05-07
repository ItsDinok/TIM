from basic_gate import task_estimation_gate, GatedResNet32, TaskEstimationGate
from torchvision import datasets, transforms
import random
from torch.utils.data import Subset
from collections import defaultdict
import torch
# TODO: Documentation everywhere

"""
List of arguments:
    - TEG -> Yes or No
    - Replay buffer -> Yes or No
    - TEG epochs -> number
    - Main model epochs -> number
    - TEG optimiser -> string
    - Main model optimiser -> string
    (TEG and main model optimiser come with parameters)
    - Dataset -> string
    - Tasks -> number
    - Classes per task -> number
    - Specific task constructions -> List of lists
    - Replay buffer size -> number
    - Replay buffer size -> string (class/task/total)
    - Buffer type -> string
    - metrics -> list of strings
    - file out -> string
    - transforms/preprocessing -> tbd
    - verbose -> yes/no

    - f -> input file (.txt)

These can be provided either as flags in the cmd or as a provided file.
-f must be used first if it is being used
"""

# ===============================================================================================
# =========================== Dataset Handlers ==================================================
# ===============================================================================================

# These handlers are just done to split the dataset into tasks that can then be returned and used
# This is done because developing tasks is one of the most tedious parts of continual learning
# TODO: Implement offbeat transforms
# TODO: Baked transform
# TODO: DIL-based handler as well as CIL
def cifar10_handler(tasks, classes_per_task, task_construction, train_ratio, verbose, random_state = None):
    if verbose:
        print("Processing CIFAR10 dataset...")

    # Fetch dataset
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )
    test = datasets.CIFAR10(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    if verbose == 2:
        print("Dataset loaded.\nCreating tasks")
    
    task_sets, task_map = create_tasks(train, verbose, tasks = tasks)
    return task_sets, task_map


def cifar100_handler(tasks, classes_per_task, task_construction, train_ratio, verbose, random_state = None):
    if verbose:
        print("Processing CIFAR100 dataset...")

    # Fetch dataset
    transform = transforms.ToTensor()
    train = datasets.CIFAR100(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )
    test = datasets.CIFAR100(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    print()


# TODO: Get this to work with different model types
def mnist_handler(tasks, classes_per_task, task_construction, train_ratio, verbose, random_state = None):
    if verbose:
        print("Processing MNIST dataset...")

    # Fetch data
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )
    mnist_test = datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    ) 

    print()


# TODO: Implement classes per task
def create_tasks(dataset, verbose, shuffle = True, tasks = None, classes_per_task = None, task_construction = None, random_state = None):
    # Get targets and classes
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    targets = torch.tensor(targets)
    classes = torch.unique(targets).tolist()

    # Determine class splits
    if task_construction is not None:
        task_classes = task_construction  
    elif classes_per_task == None:
        if tasks is None:
            raise ValueError("'tasks' must be specified")
        classes_per_task = len(classes) // tasks
        if verbose == 2:
            print(f"Classes per task: {classes_per_task}")

        # Shuffle data
        if shuffle:
            if random_state is None:
                random_state = 3407 # TODO: Allow true randomness
            random.seed(random_state)
            random.shuffle(classes)

        task_classes = [
            classes[i:i + classes_per_task]
            for i in range(0, len(classes), classes_per_task)
        ]
    
    # Map classes to indices
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    # Create subsets
    tasks = []
    task_class_map = {}
    
    for task_id, class_group in enumerate(task_classes):
        indices = []
        for cls in class_group:
            if cls in class_indices:
                indices.extend(class_indices[cls])
        tasks.append(Subset(dataset, indices))
        task_class_map[task_id] = class_group
    
    if verbose == 2:
        print("-----------")
        print(f"Created {len(task_class_map)} tasks.")
        print(f"Task map:")
        print(task_class_map)

    return tasks, task_class_map


# =================================================


def write_results(results, file_out):
    # TODO: Formatting
    with open(file_out, "a") as f:
        f.write(results)


# TODO: Range based iterative experiments
def launch_experiment(
    teg: bool = True, teg_model: torch.module = None,
    ready_model: torch.module = None,
    replay_buffer: bool = True,
    teg_epochs: int = 30, model_epochs: int = 50,
    teg_optimiser: str = "sgd", model_optimiser: str = "sgd",
    dataset: str = None,
    tasks: int = 3, classes_per_task: int = -1, # TODO: make this non-homogenous
    task_construction: list = None, 
    rb_size: int = 200, rb_size_type: str = "class", rb_type: str = "naive_lazy",
    metrics: list = None, file_out: str = None, 
    train_ratio: float = 0.7,
    verbose: int = 0
    ):
    if dataset is None:
        print("No dataset provided")
    
    # Handle dataset
    # This is where tasking happens
    if dataset.lower() == "cifar10":
        task_list, task_map = cifar10_handler(tasks, classes_per_task, task_construction, train_ratio, verbose)
    elif dataset.lower() == "cifar100":
        task_list, task_map = cifar100_handler(tasks, classes_per_task, task_construction, train_ratio, verbose)
    elif dataset.lower() == "mnist":
        task_list, task_map = mnist_handler(tasks, classes_per_task, task_construction, train_ratio, verbose)
    else:
        print(f"Dataset: {dataset} not recognised.")
        return

    if teg is True:
        if teg_model is None:
            print()
        else:
            teg = teg_model

    if teg:
        print()
    if replay_buffer:
        print()


    # TODO: Implement this
    results = ""
    print(results)

    if file_out is not None:
        write_results()
    print("Finished successfully")


# NOTE: Debug purposes only
def debug_main():
    launch_experiment(dataset = "cifar10", verbose = 2)

if __name__ == "__main__":
    debug_main()

