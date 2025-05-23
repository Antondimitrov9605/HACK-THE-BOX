import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
import joblib

# Устройство
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[i] Using device: {DEVICE}")

# ==== Dataset Class ====
class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"])
        }

# ==== Клас на модела ====
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0]
        return self.classifier(hidden)

# ==== Зареждане на данни ====
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_data("skills_assessment_data/train.json")[:30]
test_data = load_data("skills_assessment_data/test.json")[:1000]

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_dataset = IMDBDataset(train_data, tokenizer)
test_dataset = IMDBDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ==== Обучение ====
model = SentimentClassifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

print("[*] Training one epoch on 30 examples...")

model.train()
total_loss = 0
loop = tqdm(train_loader, desc="Training")
for batch in loop:
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = model(input_ids, attention_mask)
    loss = loss_fn(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    loop.set_postfix(loss=loss.item())

print(f"[✓] Training complete. Loss: {total_loss:.4f}")

# ==== Оценка ====
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"[✓] Final Accuracy: {accuracy:.4f}")

# ==== Запазване като .joblib ====
example = next(iter(test_loader))
example_input_ids = example["input_ids"].to(DEVICE)
example_attention_mask = example["attention_mask"].to(DEVICE)

traced = torch.jit.trace(model, (example_input_ids, example_attention_mask))
# Сериализирай оригиналния PyTorch модел, без TorchScript
joblib.dump(model, "skills_assessment.joblib")

print("[✓] Model saved as skills_assessment.joblib")
