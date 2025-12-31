# =========================
# REST OF PIPELINE: EMBEDDINGS + CHROMADB
# =========================
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from final import KidneyDataset  # your dataset class
import chromadb
import numpy as np

# =========================
# CONFIG
# =========================
CSV_PATH = "kidney_dataset_labels.csv"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["cyst", "stone", "tumor", "normal"]

# =========================
# TRANSFORMS
# =========================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# =========================
# LOAD DATA
# =========================
dataset = KidneyDataset(CSV_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# LOAD TRAINED MODEL
# =========================
densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 4)
densenet.load_state_dict(torch.load("trained_densenet.pth", map_location=DEVICE))
densenet.eval()
densenet = densenet.to(DEVICE)

# =========================
# 1️⃣ EXTRACT EMBEDDINGS
# =========================
def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    metadatas = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            feats = model.features(images)
            feats = torch.flatten(feats, start_dim=1)

            embeddings.extend(feats.cpu().numpy())
            metadatas.extend(labels.cpu().numpy())

    return embeddings, metadatas

embeddings, metadatas = extract_embeddings(densenet, loader)

# =========================
# 2️⃣ CONVERT ONE-HOT LABELS TO STRING METADATA
# =========================
str_metadatas = [
    {"label": CLASS_NAMES[int(np.argmax(m))]} for m in metadatas
]

# =========================
# 3️⃣ STORE EMBEDDINGS IN CHROMADB
# =========================
client = chromadb.Client()
collection = client.create_collection("kidney_embeddings")

BATCH_SIZE_CHROMA = 5000

collection.add(
    embeddings=[e.tolist() for e in embeddings],
    metadatas=str_metadatas,
    ids=[f"img_{i}" for i in range(len(embeddings))]
)

print("✅ SUCCESS: Embeddings stored in ChromaDB with string metadata!")
