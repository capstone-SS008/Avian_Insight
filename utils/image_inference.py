import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Class names must match training order
class_names = [
    'acridotheres_tristis',
    'centropus_sinensis',
    'coracias_benghalensis',
    'halcyon_smyrnensis',
    'pavo_cristatus',
    'rock_pigeon',
    'tyto_alba'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = len(class_names)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(
    torch.load("app/models/bird_img_classifier.pth", map_location=device)
)
model.to(device)
model.eval()

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

async def predict_bird_image(file):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    return class_names[pred.item()]
