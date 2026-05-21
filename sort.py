import json
import os
import shutil

with open("labels.json", "r") as f:
    data = json.load(f)

count = 0

for key in data:
    img = os.path.basename(data[key]["image_filepath"])
    cls = data[key]["anomaly_class"]

    if cls == "No-Anomaly":
        folder = "Healthy"
    elif cls == "Cracking":
        folder = "Cracked"
    elif "Hot-Spot" in cls:
        folder = "Hot-Spot"
    elif cls in ["Soiling", "Vegetation", "Shadowing"]:
        folder = "Dust-Covered"
    else:
        continue

    src = os.path.join("images", img)
    dst = os.path.join("data", "train", folder, img)

    if os.path.exists(src):
        shutil.copy(src, dst)
        count += 1

print("Copied images:", count)
