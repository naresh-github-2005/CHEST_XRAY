# app.py
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
import os
import io

st.set_page_config(page_title="Pneumonia Classifier", layout="centered")

st.title("Pneumonia classifier (Streamlit)")
st.write("Upload a chest X-ray image (jpg/png). The model predicts: Normal or Pneumonia.")

# ---------- Helper: load labels ----------
def load_labels(path="labels.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # fallback labels
    return ["Normal", "Pneumonia"]

labels = load_labels()

# ---------- Optionally download model from Google Drive ----------
# If you can't put model.pth into the repo (too large), upload it publicly to Drive
# and set DRIVE_DOWNLOAD = True and fill DRIVE_FILE_ID below.
DRIVE_DOWNLOAD = True
DRIVE_FILE_ID = "1q0AWtLtHhpAtub7HMHPG50lrkNWw0TEU"  # replace if using Drive
MODEL_PATH = "modl_best.pth"

if DRIVE_DOWNLOAD and (not os.path.exists(MODEL_PATH)):
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        st.info("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Download finished.")
    except Exception as e:
        st.error(f"Could not download model: {e}")

# ---------- Define a simple model architecture (ResNet18 commonly used) ----------
# IMPORTANT: This architecture must match how you trained your model.
# If you used a custom network, paste its class here.
import torchvision.models as models
def get_model(num_classes=len(labels)):
    model = models.resnet18(pretrained=False)
    # replace final layer
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

# ---------- Robust model loader ----------
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.warning(f"Model file '{path}' not found. Put model.pth in the app folder or enable Drive download.")
        return None

    # Try to load checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    # instantiate model
    model = get_model()
    try:
        # common shapes: either a state_dict, or a dict containing 'model_state_dict' or 'state_dict'
        if isinstance(checkpoint, dict):
            # If it contains nested keys:
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                model.load_state_dict(state)
            elif 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
                # sometimes keys start with 'module.' remove if necessary
                new_state = {}
                for k,v in state.items():
                    new_state[k.replace("module.", "")] = v
                model.load_state_dict(new_state)
            else:
                # assume this dict is a state_dict itself
                # sanitize keys if they have 'module.' prefix
                try:
                    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
                except Exception:
                    # fallback: maybe model was saved as whole object
                    model = checkpoint
        else:
            # checkpoint is not dict -> maybe full model object
            model = checkpoint
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

    model.eval()
    return model

model = load_model()

# ---------- Preprocessing ----------
def preprocess_image(img: Image.Image):
    # convert to RGB and apply transforms matching training
    img = img.convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # 1 x C x H x W

# ---------- UI: upload ----------
uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg","jpeg","png"])

# Example quick-predict button using a sample image in repo (optional)
st.write("Or try the sample image below (if present):")
sample_button = st.button("Use sample image (sample.jpg)")

image = None
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
elif sample_button:
    if os.path.exists("sample.jpg"):
        image = Image.open("sample.jpg")
    else:
        st.warning("No sample.jpg found in the app folder.")

if image is not None:
    st.image(image, caption="Input image", use_column_width=True)
    if model is None:
        st.error("Model not loaded. See earlier messages.")
    else:
        with st.spinner("Running prediction..."):
            x = preprocess_image(image)
            with torch.no_grad():
                outputs = model(x)
                # If outputs is a tensor or dict etc.
                if isinstance(outputs, torch.Tensor):
                    logits = outputs.squeeze(0)
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[0].squeeze(0)
                else:
                    # fallback
                    logits = torch.tensor(outputs).squeeze(0)

                probs = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
                top_idx = int(np.argmax(probs))
                top_prob = float(probs[top_idx])
                pred_label = labels[top_idx] if top_idx < len(labels) else str(top_idx)

        st.markdown("### Prediction")
        st.write(f"**{pred_label}** — confidence: **{top_prob*100:.2f}%**")

        # Show full probabilities table
        st.markdown("#### All class probabilities")
        for i,c in enumerate(labels):
            st.write(f"{c}: {probs[i]*100:.2f}%")



