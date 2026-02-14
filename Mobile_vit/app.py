import gradio as gr
import torch
import timm
import json
from torchvision import transforms
from PIL import Image

# Device
DEVICE = "cpu"

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# Load model
model = timm.create_model(
    "mobilevit_s",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load("mobilevit_compressed_best.pth", map_location=DEVICE)
)

model.eval()
model.to(DEVICE)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict(image):

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)[0]

    return {
        class_names[i]: float(probs[i])
        for i in range(NUM_CLASSES)
    }

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="MobileViT Waste Classifier",
    description="Upload waste image for classification"
)

interface.launch()
