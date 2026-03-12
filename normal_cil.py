import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import copy
import numpy as np


class ExemplarDataset(Dataset):
    """
    Dataset wrapper for exemplar memory.
    Stores raw images and labels.
    """
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            if isinstance(x, torch.Tensor):
                x = x
            else:
                x = self.transform(x)

        return x, y


class CosineClassifier(nn.Module):
    """
    Cosine similarity classifier.
    Used by iCaRL, LUCIR, etc
    """
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.normal_(self.weight, 0, 0.01)

    def forward(self, x):
        x = F.normalize(x, dim = 1)
        w = F.normalize(self.weight, dim = 1)

        return x @ w.t()


class IncrementalResNet(nn.Module):
    """
    ResNet-based Class Incremental Learning model using:
    LwF (Learning without forgetting)
    iCaRL (Incremental classifier and representation learning)
    """
    def __init__(self, feature_dim = 512, device = "cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.seen_classes = 0
        
        # Backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        backbone.maxpool = nn.Identity()

        modules = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        self.classifier = None
        self.old_model = None

        # Exemplar memory
        self.exemplar_images = []
        self.exemplar_labels = []

        # Freeze batchnorm
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.to(device)

    def forward(self, x):
        features = self.extract_features(x)
        if self.classifier is None:
            return features

        return self.classifier(features)

    def extract_features(self, x):
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        return f

    def increment_classes(self, n_new):
        """
        Expand classifier to accommodate new classes
        """
        old_weights = None

        if self.classifier is not None:
            old_weights = self.classifier.weight.data
        
        new_classes = self.seen_classes + n_new
        new_classifier = CosineClassifier(self.feature_dim, new_classes).to(self.device)

        if old_weights is not None:
            new_classifier.weight.data[:self.seen_classes] = old_weights

        self.classifier = new_classifier
        self.seen_classes = new_classes

    def set_old_model(self):
        """
        Keep a frozen copy of the current model for LwF
        """
        self.old_model = copy.deepcopy(self).eval()
        for param in self.old_model.parameters():
            param.requires_grad = False

    def distillation_loss(self, inputs, targets, T = 2, alpha = 0.1):
        logits = self(inputs)
        ce = F.cross_entropy(logits, targets)

        if self.old_model is None:
            return ce

        with torch.no_grad():
            old_logits = self.old_model(inputs)
            old_probs = F.softmax(old_logits / T, dim = 1)

        kd = F.kl_div(
                F.log_softmax(logits[:, :old_logits.size(1)] / T, dim = 1),
                old_probs,
                reduction = "batchmean") * (T * T)
        return alpha * ce + (1 - alpha) * kd

    def build_exemplars(self, dataset, m):
        """
        Herding based exemplar selection per class
        """
        self.feature_extractor.eval()
        class_dict = {}

        # Collect indices per class
        for idx, (_, y) in enumerate(dataset):
            class_dict.setdefault(y, []).append(idx)

        # Compute features for all data once
        data_loader = DataLoader(dataset, batch_size = 128, shuffle = False)
        features_all = []
        labels_all = []
        for x, y in data_loader:
            x = x.to(self.device)
            with torch.no_grad():
                f = self.extract_features(x)
                f = F.normalize(f, dim = 1)
            features_all.append(f.cpu())
            labels_all.append(y)
        features_all = torch.cat(features_all)

        new_images = []
        new_labels = []

        # Herding selection per class
        for cls, indices in class_dict.items():
            f_cls = features_all[indices]
            f_mean = f_cls.mean(0)
            selected = []
            sum_f = torch.zeros_like(f_mean)
           
            available = torch.ones(len(f_cls), dtype = torch.bool)
            for _ in range(min(m, len(f_cls))):
                k = len(selected) + 1
                distances = ((f_mean - (sum_f + f_cls) / k) ** 2).sum(dim = 1)
            
                distances[~available] = float("inf")
            
                idx = distances.argmin().item()
                selected.append(indices[idx])
                sum_f += f_cls[idx]
                available[idx] = False

            # Store raw images
            base_dataset = dataset.dataset

            for idx in selected:
                real_idx = dataset.indices[idx]
                img, _ = base_dataset[real_idx]
                new_images.append(img)
                new_labels.append(cls)

        self.exemplar_images.extend(new_images)
        self.exemplar_labels.extend(new_labels)

    def reduce_exemplars(self, m):
        """
        Reduce stored exemplars so each class has at most m samples
        """
        new_images = []
        new_labels = []

        class_dict = {}

        for img, label in zip(self.exemplar_images, self.exemplar_labels):
            class_dict.setdefault(label, []).append(img)

        for cls, imgs in class_dict.items():
            imgs = imgs[:m] # Keep only first m
            new_images.extend(imgs)
            new_labels.extend([cls] * len(imgs))

        self.exemplar_images = new_images
        self.exemplar_labels = new_labels

    def compute_class_means(self):
        """
        Compute mean feature for each class from exemplars
        """
        self.eval()

        class_dict = {}

        for img, label in class_dict.items():
            class_dict.setdefault(label, []).append(img)

        class_means = {}

        with torch.no_grad():
            for cls, imgs in class_dict.items():
                xs = torch.stack(imgs).to(self.device)
                feats = self.extract_features(xs)
                feats = F.normalize(feats, dim = 1)

                mean = feats.mean(0)
                mean = F.normalize(mean, dim = 0)

                class_means[cls] = mean

        return class_means

    def ncm_predict(self, x, class_means):
        """
        Nearest Class Mean prediction
        """
        feats = self.extract_features(x)
        feats = F.normalize(feats, dim = 1)

        means = torch.stack([class_means[c] for c in sorted(class_means.keys())])
        means = means.to(self.device)

        dists = torch.cdist(feats, means)
        preds = dists.argmin(dim = 1)

        return preds


def evaluate(model, dataloader, device = "cuda"):
    model.eval()
    
    correct = 0
    total = 0

    class_means = model.compute_class_means()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if len(class_means) == 0:
                logits = model(x)
                preds = logits.argmax(1)
            else:
                preds = model.ncm_predict(x, class_means)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train_task(model, loader, batch_size, epochs, lr, device = "cuda"):
    # Prepare replay loader if exemplars exist
    if model.exemplar_images:
        replay_dataset = ExemplarDataset(model.exemplar_images, model.exemplar_labels, transform = loader.dataset.dataset.transform)
        replay_loader = DataLoader(replay_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    else:
        replay_loader = None


    # Freeze early layers, only train last block
    for p in model.feature_extractor.parameters():
        p.requires_grad = True

    optimiser = torch.optim.Adam([
        {"params": model.feature_extractor.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": lr}
    ])

    model.train()
    for epoch in range(epochs):
        new_iter = iter(loader)
        if replay_loader is not None:
            replay_iter = iter(replay_loader)
        else:
            replay_iter = None
        
        while True:
            try:
                x_new, y_new = next(new_iter)
            except StopIteration:
                break

            x_new = x_new.to(device)
            y_new = y_new.to(device)

            # Combine with replay batch
            if replay_loader is not None and replay_iter is not None:
                try:
                    x_old, y_old = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    x_old, y_old = next(replay_iter)
                
                x_old = x_old.to(device)
                y_old = y_old.to(device)

                x = torch.cat((x_new, x_old), dim = 0)
                y = torch.cat((y_new, y_old), dim = 0)
                replay_iter = iter(replay_loader) # shuffle replay loader
            else:
                x = x_new
                y = y_new

            optimiser.zero_grad()
            loss = model.distillation_loss(x, y)
            loss.backward()
            optimiser.step()


def compute_bwt(R):
    T = R.shape[0]
    bwt = 0
    
    for i in range(T - 1):
        bwt += R[T - 1, i] - R[i, i]

    bwt /= (T - 1)
    return bwt


def compute_fwt(R, baseline = None):
    T = R.shape[0]
    fwt = 0

    for i in range(1, T):
        if baseline is None:
            b = 0
        else:
            b = baseline[i]

        fwt += R[i - 1, i] - b
    fwt /= (T - 1)
    return fwt


def build_tasks(num_tasks = 10, classes_per_task = 10):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    task_train_datasets = []
    task_test_datasets = []

    for task_id in range(num_tasks):

        start = task_id * classes_per_task
        end = start + classes_per_task

        # noinspection PyTypeChecker
        train_indices = [
            i for i, (_, y) in enumerate(train_dataset)
            if start <= y < end
        ]

        # noinspection PyTypeChecker
        test_indices = [
            i for i, (_, y) in enumerate(test_dataset)
            if start <= y < end
        ]

        task_train_datasets.append(
            Subset(train_dataset, train_indices)
        )

        task_test_datasets.append(
            Subset(test_dataset, test_indices)
        )

    return task_train_datasets, task_test_datasets


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    num_tasks = 5
    classes_per_task = 2
    epochs = 15
    batch_size = 128
    lr = 1e-3
    total_memory = 2000

    # Model
    model = IncrementalResNet(device = device)

    # Data
    train_tasks, test_tasks = build_tasks(num_tasks, classes_per_task)

    # Accuracy matrix
    R = np.zeros((num_tasks, num_tasks))

    # CL Loop
    for task in range(num_tasks):
        print(f"\n=== TASK {task} ===")
        model.increment_classes(classes_per_task)
        train_loader = DataLoader(train_tasks[task], batch_size=batch_size, shuffle=True, drop_last = True)

        # Train task with replay
        train_task(model, train_loader, batch_size, epochs, lr, device)

        # Evaluate all seen tasks
        for t in range(task+1):
            test_loader = DataLoader(test_tasks[t], batch_size=256)
            acc = evaluate(model, test_loader, device)
            R[task, t] = acc
            print(f"Eval Task {t} Acc: {acc:.4f}")

        # Update exemplars with herding
        m = total_memory // model.seen_classes
        model.reduce_exemplars(m)
        model.build_exemplars(train_tasks[task], m)
        # Save old model for distillation
        model.set_old_model()
    
    # Metrics
    bwt = compute_bwt(R)
    fwt = compute_fwt(R)

    print("\n========== RESULTS ==========")
    print("Accuracy Matrix:\n", R)
    print("Backward Transfer (BwT):", bwt)
    print("Forward Transfer (FwT):", fwt)


if __name__ == "__main__":
    main()
