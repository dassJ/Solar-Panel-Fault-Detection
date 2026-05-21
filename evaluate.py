import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/train", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

classes = dataset.classes

# Load trained ResNet18
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("solar_model.pth", map_location="cpu"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=classes))

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()