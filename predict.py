import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

classes = ['Cracked', 'Dust-Covered', 'Healthy', 'Hot-Spot']

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("solar_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_path = "images/13357.jpg"   # change image here
img = Image.open(img_path).convert("RGB")

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    _, pred = torch.max(output, 1)

pred_class = classes[pred.item()]
max_conf = probs[0][pred.item()].item() * 100

print("\n--- Prediction Details ---")
for i, cls in enumerate(classes):
    print(f"{cls}: {probs[0][i].item() * 100:.2f}%")

print("\nFinal Prediction:", pred_class)
print(f"Confidence: {max_conf:.2f}%")

# Alert system
if pred_class in ["Cracked", "Hot-Spot"]:
    print("ALERT: Maintenance Required")
elif pred_class == "Dust-Covered":
    print("WARNING: Cleaning Recommended")
else:
    print("Status: Panel is Healthy")

# Uncertain prediction logic
if max_conf < 60:
    print("Decision: Uncertain prediction")
else:
    print("Decision: Confident prediction")