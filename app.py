import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st

classes = ['Cracked', 'Dust-Covered', 'Healthy', 'Hot-Spot']

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("solar_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("Solar Panel Fault Detection")
st.write("Upload a thermal image to classify the panel condition.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    pred_class = classes[pred.item()]
    max_conf = probs[0][pred.item()].item() * 100

    st.subheader("Final Prediction")
    st.write(pred_class)

    st.subheader("Confidence Score")
    st.write(f"{max_conf:.2f}%")

    st.subheader("Confidence Scores for All Classes")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[0][i].item() * 100:.2f}%")

    if pred_class in ["Cracked", "Hot-Spot"]:
        st.error("ALERT: Maintenance Required")
    elif pred_class == "Dust-Covered":
        st.warning("Cleaning Recommended")
    else:
        st.success("Panel is Healthy")

    if max_conf < 60:
        st.info("Prediction is uncertain. Check manually.")