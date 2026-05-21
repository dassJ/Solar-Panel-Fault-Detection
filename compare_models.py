import os
import time
import torch
from torchvision import models, transforms
from PIL import Image

classes = ['Cracked', 'Dust-Covered', 'Healthy', 'Hot-Spot']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("images/13357.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)

# ResNet
resnet = models.resnet18()
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 4)
resnet.load_state_dict(torch.load("solar_model.pth", map_location="cpu"))
resnet.eval()

start = time.time()
with torch.no_grad():
    out1 = resnet(img_tensor)
resnet_time = time.time() - start
resnet_pred = classes[out1.argmax(dim=1).item()]
resnet_size = os.path.getsize("solar_model.pth") / (1024 * 1024)

# MobileNet
mobilenet = models.mobilenet_v3_small()
mobilenet.classifier[3] = torch.nn.Linear(mobilenet.classifier[3].in_features, 4)
mobilenet.load_state_dict(torch.load("mobilenet_model.pth", map_location="cpu"))
mobilenet.eval()

start = time.time()
with torch.no_grad():
    out2 = mobilenet(img_tensor)
mobilenet_time = time.time() - start
mobilenet_pred = classes[out2.argmax(dim=1).item()]
mobilenet_size = os.path.getsize("mobilenet_model.pth") / (1024 * 1024)

print("\n--- Model Comparison ---")
print(f"ResNet Prediction: {resnet_pred}")
print(f"ResNet Inference Time: {resnet_time:.4f} sec")
print(f"ResNet Model Size: {resnet_size:.2f} MB")

print(f"\nMobileNet Prediction: {mobilenet_pred}")
print(f"MobileNet Inference Time: {mobilenet_time:.4f} sec")
print(f"MobileNet Model Size: {mobilenet_size:.2f} MB")

print("\nConclusion:")
if resnet_time > mobilenet_time:
    print("MobileNet is faster and better for lightweight deployment.")
else:
    print("ResNet is faster on this system.")

if resnet_size > mobilenet_size:
    print("MobileNet uses less storage.")
else:
    print("ResNet uses less storage.")