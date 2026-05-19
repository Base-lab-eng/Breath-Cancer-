
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import io

from Transfer_Learning import build_model
from utils_py import load_checkpoint

# ─────────────────────────────────────────────
app = FastAPI(title="Breast Cancer Classification API")

# ── Device ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Classes ──────────────────────────────────
CLASSES = ["benign", "malignant"]


# ── Load Model ───────────────────────────────
model = build_model()


path=r"C:\Users\basem\Desktop\Final project\checkpoints\best_model_resnet50.pth"
epoch = load_checkpoint(path, model)  

model.to(device)
model.eval()



# ── Preprocessing ────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Breast Cancer Classification API",
        "model": "ResNet-50",
        "classes": CLASSES
    }

# ─────────────────────────────────────────────
@app.post("/classify")
async def classify(file: UploadFile = File(...)):

    # ── Validate image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    except Exception:
        return {
            "error": "Invalid image file"
        }

    # ── Preprocessing
    tensor = transform(image).unsqueeze(0).to(device)

    # ── Inference
    with torch.inference_mode():

        outputs = model(tensor)

        probs = torch.softmax(outputs, dim=1)

        conf, pred = probs.max(1)

    prediction = CLASSES[pred.item()]
    confidence = round(conf.item(), 4)

    print(f"Prediction: {prediction} | Confidence: {confidence}")

    return {
        "prediction": prediction,
        "confidence": confidence
    }

