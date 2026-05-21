import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
img_resized = img.resize((224, 224))
img_tensor = transform(img).unsqueeze(0)

features = None
gradients = None

def forward_hook(module, inp, out):
    global features
    features = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer = model.layer4[1].conv2

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

output = model(img_tensor)
pred_class_idx = output.argmax(dim=1).item()
pred_class = classes[pred_class_idx]

model.zero_grad()
output[0, pred_class_idx].backward()

pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
feature_maps = features[0].detach().clone()

for i in range(feature_maps.shape[0]):
    feature_maps[i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(feature_maps, dim=0).numpy()
heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (heatmap.max() + 1e-8)

img_np = np.array(img_resized)
img_np = cv2.resize(img_np, (224, 224))

heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_uint8 = cv2.resize(heatmap_uint8, (224, 224))
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

# Severity estimation
threshold = 0.6
fault_pixels = np.sum(heatmap > threshold)
total_pixels = heatmap.shape[0] * heatmap.shape[1]
severity_percent = (fault_pixels / total_pixels) * 100

if severity_percent < 5:
    severity_level = "Low"
elif severity_percent < 15:
    severity_level = "Medium"
else:
    severity_level = "High"

# Simple power loss estimate
power_loss = severity_percent * 0.5

plt.figure(figsize=(6, 6))
plt.imshow(img_np)
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title(f"Overlay - Prediction: {pred_class}")
plt.axis("off")
plt.show()

print("Predicted class:", pred_class)
print(f"Fault Severity Area: {severity_percent:.2f}%")
print("Severity Level:", severity_level)
print(f"Estimated Power Loss: {power_loss:.2f}%")

# Alert logic
if pred_class in ["Cracked", "Hot-Spot"] and severity_level in ["Medium", "High"]:
    print("ALERT: Immediate Maintenance Required")
elif pred_class == "Dust-Covered":
    print("NOTICE: Cleaning Recommended")
else:
    print("Status: Normal / Low Risk")

forward_handle.remove()
backward_handle.remove()