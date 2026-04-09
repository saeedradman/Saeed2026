import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from dataset import MRIDataset
from model import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
def load_data(split):
    X, y = [], []
    for label, cls in enumerate(["control", "ms"]):
        folder = f"data/processed/{split}/{cls}"
        for img in os.listdir(folder):
            X.append(os.path.join(folder, img))
            y.append(label)
    return X, y

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3),
])

X_val, y_val = load_data("val")
val_ds = MRIDataset(X_val, y_val, val_tf)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

# Load Model
model = CNN().to(device)
model.load_state_dict(torch.load("models/cnn_ms_best.pth", map_location=device))
model.eval()

# Inference
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_dl:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Metrics
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Control", "MS"]
))

cm = confusion_matrix(y_true, y_pred)

# Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Control", "MS"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - CNN MS Classifier")
plt.show()