import torch
import torch.nn as nn
import librosa
import numpy as np
from PIL import Image
from timm import create_model
from torchvision import transforms as T
import matplotlib.pyplot as plt
import io

# Same 5 classes as used during training
selected_birds = [
    "Acridotheres_tristis",
    "Halcyon_smyrnensis",
    "Pavo_cristatus",
    "Centropus_sinensis",
    "Coracias_benghalensis"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT/AST audio classifier
model = create_model("vit_base_patch16_224", pretrained=True, num_classes=len(selected_birds))
model.head = nn.Linear(model.head.in_features, len(selected_birds))
model.load_state_dict(torch.load("bird_models/ast_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Transform (same as training)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

def audio_to_mel_spectrogram(audio_bytes, sr=16000, n_mels=128):
    # Convert audio bytes → waveform
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

async def predict_bird_sound(file):
    audio_bytes = await file.read()

    mel = audio_to_mel_spectrogram(audio_bytes)

    # Save mel-spec as temp image
    plt.imsave("temp.png", mel, cmap="viridis")

    img = Image.open("temp.png").convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    return selected_birds[pred.item()]
