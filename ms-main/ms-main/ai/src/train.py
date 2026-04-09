# MRI Multiple Sclerosis Classification (CNN from scratch)


import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import MRIDataset
from model import CNN
import torch 


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Ensure directories exist
os.makedirs("models", exist_ok=True)


# Data Loading Function
def load_data(split):
    X, y = [], []
    for label, cls in enumerate(["control", "ms"]):
        folder = f"data/processed_clean/{split}/{cls}"
        for img in os.listdir(folder):
            if img.lower().endswith((".png", ".jpg", ".jpeg")):
                X.append(os.path.join(folder, img))
                y.append(label)
    return X, y


# Data Augmentation (MRI-Safe & Conservative)
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # MRI-safe
    transforms.RandomRotation(7),             # subtle rotation
    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],   # Better for CNN from scratch
        std=[0.5, 0.5, 0.5]
    ),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# Load Datasets
X_train, y_train = load_data("train")
X_val, y_val = load_data("val")

train_ds = MRIDataset(X_train, y_train, train_tf)
val_ds = MRIDataset(X_val, y_val, val_tf)

train_dl = DataLoader(
    train_ds,
    batch_size=8,      # Small batch for MRI stability
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_dl = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")


# Model
model = CNN().to(device)
print("Model initialized.")


# Loss Function (Medical-aware)
class_weights = torch.tensor([1.0, 1.4]).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1   # reduces overconfidence
)


# Optimizer & Scheduler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-3
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.3,
    patience=3,
)


# Training Configuration
num_epochs = 40
best_acc = 0.0

early_stop_patience = 6
early_stop_counter = 0


# Training Loop
for epoch in range(num_epochs):

    # Training
    model.train()
    train_loss = 0.0

    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping (important for deep CNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dl)

    # Validation
    model.eval()
    val_loss = 0.0
    preds, true = [], []

    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds.extend(outputs.argmax(1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    val_loss /= len(val_dl)
    val_acc = accuracy_score(true, preds)

    # Logging
    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # Scheduler
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current LR: {current_lr:.6f}")
    # Early Stopping & Saving
    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0

        torch.save(
            model.state_dict(),
            "models/cnn_ms_best.pth"
        )
        print("Best model saved.")

    else:
        early_stop_counter += 1
        print(
            f"No improvement "
            f"({early_stop_counter}/{early_stop_patience})"
        )

    if early_stop_counter >= early_stop_patience:
        print("Early stopping triggered.")
        break


# Save Last Model
torch.save(
    model.state_dict(),
    "models/cnn_ms_last.pth"
)

print("Last model saved.")
print("Training completed.")
print("Best Validation Accuracy:", best_acc)