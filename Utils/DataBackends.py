from abc import ABC, abstractmethod
import get_mnist

# TODO: Add docs here
# TODO: Standardise get_mnist, it seems very aetherial rn

class DataBackend(ABC):
    @abstractmethod
    def load(self, root):
        pass

    @abstractmethod
    def build_tasks(self, root, train_transform, test_transform):
        pass

    @abstractmethod
    def split_gate(self, dataset):
        pass

    @abstractmethod
    def prepare_gate(self, tasks):
        pass


class MNISTBackend(DataBackend):
    def load(self, root):
        return get_mnist.load_mnist(root = root)

    def build_tasks(self, root, train_transform, test_transform):
        return get_mnist.build_mnist_cil_tasks(
            root = root,
            train_transform = train_transform,
            test_transform = test_transform,
            use_local_labels = False,
            download = True
        )

    def split_gate(self, dataset):
        return get_mnist.split_dataset_by_tasks(dataset = dataset)

    def prepare_gate(self, tasks):
        return get_mnist.prepare_gate_datasets(tasks)