# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import chromadb
import numpy as np

# =========================
# CONFIG
# =========================
CSV_PATH = "kidney_dataset_labels.csv"
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class KidneyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)

        # Normalize column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]

        self.label_cols = ["cyst", "stone", "tumor", "normal"]

        # Ensure numeric labels
        for col in self.label_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": "0"})
                .astype(float)
            )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["imagen"]
        image = Image.open(img_path).convert("RGB")

        label = self.df.iloc[idx][self.label_cols].to_numpy(dtype=np.float32)
        label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# TRANSFORMS & LOADER
# =========================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = KidneyDataset(CSV_PATH, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL
# =========================
densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 4)
densenet = densenet.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=1e-4)

# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    densenet.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = densenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# =========================
# EMBEDDINGS EXTRACTION
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
# CONVERT LABELS TO STRING METADATA
# =========================
CLASS_NAMES = ["cyst", "stone", "tumor", "normal"]

str_metadatas = [
    {"label": CLASS_NAMES[int(m.argmax())]} for m in metadatas
]

# =========================
# STORE IN CHROMADB
# =========================
client = chromadb.Client()
collection = client.create_collection("kidney_embeddings")

BATCH_SIZE_CHROMA = 5000

collection.add(
    embeddings=[e.tolist() for e in embeddings],
    metadatas=str_metadatas,
    ids=[f"img_{i}" for i in range(len(embeddings))]
)

print("✅ SUCCESS: Training finished & embeddings stored in ChromaDB")
