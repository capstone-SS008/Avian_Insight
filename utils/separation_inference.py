import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
from PIL import Image
from timm import create_model
from asteroid.models import ConvTasNet
from torchvision import transforms
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD SEPARATOR MODEL
# -----------------------------
separator = ConvTasNet(n_src=2)
separator.load_state_dict(torch.load(
    "backend/bird_models/conv_tasnet_mixed_classifier.pth",
    map_location=device
))
separator.to(device)
separator.eval()

# -----------------------------
# LOAD CLASSIFIER MODEL
# -----------------------------
selected_birds = [
    "Acridotheres_tristis",
    "Halcyon_smyrnensis",
    "Pavo_cristatus",
    "Centropus_sinensis",
    "Coracias_benghalensis"
]

classifier = create_model("vit_base_patch16_224", pretrained=True, num_classes=len(selected_birds))
classifier.head = nn.Linear(classifier.head.in_features, len(selected_birds))

classifier.load_state_dict(torch.load(
    "bird_models/ast_classifier.pth",
    map_location=device
))
classifier.to(device)
classifier.eval()

# -----------------------------
# AUDIO LOADING
# -----------------------------
def load_audio(path, target_sr=8000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = torch.tensor(audio, dtype=torch.float32)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr

    return audio, sr

# -----------------------------
# SEPARATION
# -----------------------------
def separate_sources(audio_path, sample_rate=8000):
    waveform, sr = load_audio(audio_path, sample_rate)
    mixture = waveform.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        est_sources = separator(mixture)

    est_sources = est_sources.squeeze(0).cpu()
    return est_sources, sample_rate

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# MEL SPECTROGRAM
# -----------------------------
def audio_to_mel_spectrogram(waveform, sr=8000, n_mels=128):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels
    )(waveform)

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    return mel_db.numpy()

# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify_waveform(waveform, sr):
    mel_spec = audio_to_mel_spectrogram(waveform, sr)
    
    temp_path = "backend/temp/temp_spec.png"
    plt.imsave(temp_path, mel_spec[0], cmap="viridis")

    img = Image.open(temp_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier(img)
        pred = torch.argmax(out, dim=1).item()

    return selected_birds[pred]

# -----------------------------
# END-TO-END FUNCTION
# -----------------------------
def separate_and_classify(audio_path):
    sources, sr = separate_sources(audio_path)

    results = []
    for idx, src in enumerate(sources):
        bird = classify_waveform(src.unsqueeze(0), sr)
        results.append({
            "source": idx + 1,
            "predicted_bird": bird
        })

    return results
