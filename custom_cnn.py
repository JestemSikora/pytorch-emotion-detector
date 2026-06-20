import torch
import torch_directml
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader, Subset, Dataset

from config import DATA_PATH, TEST_PATH

EPOCHS = 60
IMG_SIZE = 48
MINORITY_CLASSES = {"disgusted"}

if torch_directml.is_available():
    device = torch_directml.device(0)
    print(f"Using DirectML Device: {torch_directml.device_name(0)}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


def get_base_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_minority_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class SelectiveAugDataset(Dataset):
    def __init__(self, roots, minority_transform, base_transform):
        if isinstance(roots, str):
            roots = [roots]

        bases = [datasets.ImageFolder(root=r, transform=None) for r in roots]
        self._loader      = bases[0].loader
        self.classes      = bases[0].classes
        self.class_to_idx = bases[0].class_to_idx

        self.samples = []
        self.targets = []
        for base in bases:
            self.samples.extend(base.samples)
            self.targets.extend(base.targets)

        minority_idx = {
            idx for name, idx in self.class_to_idx.items()
            if name in MINORITY_CLASSES
        }
        self._minority_idx = minority_idx
        self._minority_tf  = minority_transform
        self._base_tf      = base_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self._loader(path)
        if label in self._minority_idx:
            img = self._minority_tf(img)
        else:
            img = self._base_tf(img)
        return img, label


class CustomCNN(nn.Module):
    """
    Architektura wzorowana na Kaggle (m0hamedreda/emotion-detector),
    przepisana na PyTorch. Wejście: 1×48×48 (grayscale).
    Po 4x MaxPool(2): 512×3×3 = 4608 przed klasyfikatorem.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


def train_and_evaluate(model, train_loader, test_loader, class_weights, epochs):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # weight_decay = L2 regularization odpowiednik kernel_regularizer=l2() z Keras
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epoch_losses, epoch_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        epoch_losses.append(running_loss / total)
        epoch_accs.append(correct / total)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc     = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    prec    = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec     = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm      = confusion_matrix(all_labels, all_preds)
    return acc, bal_acc, prec, rec, cm, epoch_losses, epoch_accs


def run_experiment(train_dataset, test_dataset, epochs):
    num_classes = len(train_dataset.classes)
    accuracies, balanced_accuracies, precisions, recalls, confusion_matrices = [], [], [], [], []
    all_epoch_losses, all_epoch_accs = [], []

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(
        rskf.split(np.arange(len(train_dataset)), train_dataset.targets)
    ):
        train_subset = Subset(train_dataset, train_idx)
        test_subset  = Subset(test_dataset,  test_idx)

        train_targets = torch.tensor([train_dataset.targets[i] for i in train_idx])
        class_counts  = torch.bincount(train_targets, minlength=num_classes).float()
        class_weights = (len(train_idx) / (num_classes * class_counts)).to(device)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_subset,  batch_size=64, shuffle=False)

        model = CustomCNN().to(device)
        acc, bal_acc, prec, rec, cm, e_losses, e_accs = train_and_evaluate(
            model, train_loader, test_loader, class_weights, epochs
        )

        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)
        precisions.append(prec)
        recalls.append(rec)
        confusion_matrices.append(cm)
        all_epoch_losses.append(e_losses)
        all_epoch_accs.append(e_accs)
        print(f"Fold {fold + 1:2d}: Accuracy={acc:.4f}, Balanced Accuracy={bal_acc:.4f}, "
              f"Precision={prec:.4f}, Recall={rec:.4f}")

    mean_epoch_losses = np.mean(all_epoch_losses, axis=0)
    mean_epoch_accs   = np.mean(all_epoch_accs,   axis=0)
    return accuracies, balanced_accuracies, precisions, recalls, confusion_matrices, mean_epoch_losses, mean_epoch_accs


def plot_training_curves(epoch_losses, epoch_accs):
    epochs = range(1, len(epoch_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, epoch_losses)
    ax1.set_title('Training Loss vs Epoch (średnia po foldach)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(epochs, epoch_accs)
    ax2.set_title('Training Accuracy vs Epoch (średnia po foldach)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(bal_accs, cms, class_names):
    mean_cm      = np.mean(cms, axis=0)
    mean_cm_norm = mean_cm / mean_cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        mean_cm_norm,
        annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_title(f'Macierz pomylek: CustomCNN\nSrednia Balanced Accuracy: {np.mean(bal_accs):.4f}')
    ax.set_xlabel('Predykcja')
    ax.set_ylabel('Rzeczywista klasa')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(cms, class_names):
    mean_cm   = np.mean(cms, axis=0)
    recall    = np.diag(mean_cm) / (mean_cm.sum(axis=1) + 1e-8)
    precision = np.diag(mean_cm) / (mean_cm.sum(axis=0) + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precision, w, label='Precision')
    ax.bar(x,     recall,    w, label='Recall')
    ax.bar(x + w, f1,        w, label='F1')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Per-class metrics: CustomCNN')
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_error_analysis(cms, class_names):
    mean_cm      = np.mean(cms, axis=0)
    mean_cm_norm = mean_cm / (mean_cm.sum(axis=1, keepdims=True) + 1e-8)

    errors = [
        (mean_cm_norm[i, j], class_names[i], class_names[j])
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j
    ]
    errors.sort(reverse=True)

    print("\n=== CustomCNN — najczęstsze błędy klasyfikacji ===")
    print(f"{'Prawdziwa':<12} → {'Predykcja':<12}  {'Częstość':>8}")
    for rate, true_cls, pred_cls in errors[:5]:
        print(f"{true_cls:<12} → {pred_cls:<12}  {rate:>8.2%}")


if __name__ == '__main__':
    roots         = [DATA_PATH, TEST_PATH]
    minority_tf   = get_minority_transform()
    base_tf       = get_base_transform()
    train_dataset = SelectiveAugDataset(roots, minority_tf, base_tf)
    test_dataset  = SelectiveAugDataset(roots, base_tf,     base_tf)

    print(f'Zaladowano obrazow: {len(train_dataset)}')
    print(f'Klasy: {train_dataset.classes}')

    accs, bal_accs, precs, recs, cms, epoch_losses, epoch_accs = run_experiment(
        train_dataset, test_dataset, EPOCHS
    )

    print(f'\n=== Wyniki CustomCNN ===')
    print(f'Srednia Accuracy:          {np.mean(accs):.4f}')
    print(f'Srednia Balanced Accuracy: {np.mean(bal_accs):.4f}')
    print(f'Srednia Precision:         {np.mean(precs):.4f}')
    print(f'Srednia Recall:            {np.mean(recs):.4f}')

    plot_training_curves(epoch_losses, epoch_accs)
    plot_confusion_matrix(bal_accs, cms, train_dataset.classes)
    plot_per_class_metrics(cms, train_dataset.classes)
    print_error_analysis(cms, train_dataset.classes)
