import torch
import os
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time

# === Клас за класификация ===
class MalwareClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MalwareClassifier, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, n_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# === Функции за трансформация и зареждане на данни ===
def load_datasets(base_path, train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.Resize((75, 75)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=os.path.join(base_path, "train"), transform=transform)
    test_dataset = ImageFolder(root=os.path.join(base_path, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    n_classes = len(train_dataset.classes)
    return train_loader, test_loader, n_classes

# === Тренировка ===
def compute_accuracy(correct, total):
    return round(100 * correct / total, 2)

def train(model, train_loader, n_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
        acc = compute_accuracy(correct, total)
        print(f"[{epoch+1}/{n_epochs}] Accuracy: {acc:.2f}% Loss: {running_loss/len(train_loader):.4f}")
    return

# === Запазване ===
def save_model(model, path):
    model_scripted = torch.jit.script(model)
    model_scripted.save(path)
    print(f"[✓] Model saved to: {path}")

# === Оценка ===
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = compute_accuracy(correct, total)
    return acc

# === СТАРТ — само ако скриптът е главен ===
if __name__ == "__main__":
    DATA_PATH = "./newdata/"
    N_EPOCHS = 10
    TRAINING_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 1024
    MODEL_FILE = "malware_classifier.pth"

    print("[*] Loading data...")
    train_loader, test_loader, n_classes = load_datasets(DATA_PATH, TRAINING_BATCH_SIZE, TEST_BATCH_SIZE)

    print("[*] Initializing model...")
    model = MalwareClassifier(n_classes)

    print("[*] Starting training...")
    train(model, train_loader, N_EPOCHS)

    save_model(model, MODEL_FILE)

    acc = evaluate(model, test_loader)
    print(f"[✓] Final Test Accuracy: {acc}%")
