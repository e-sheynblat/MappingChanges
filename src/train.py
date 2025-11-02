import random
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import ChangeDetectionDataset
from siamese_model import ResNet50Siamese
from fetch_historical import fetch_gee_tile
from fetch_current import fetch_google_tile
import os

# ---------------------- Contrastive Loss ----------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distance = torch.sqrt(torch.sum((out1 - out2) ** 2, dim=1) + 1e-6)
        loss = 0.5 * (
            label * distance ** 2
            + (1 - label) * torch.clamp(self.margin - distance, min=0.0) ** 2
        )
        return loss.mean()


# ---------------------- Dataset (uses prefetched tiles) ----------------------
class ContrastiveDataset(Dataset):
    def __init__(self, fetched_tiles, transform=None):
        self.tiles = fetched_tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        lat1, lon1, hist1, curr1 = self.tiles[idx]

        # Positive or negative pair
        if random.random() < 0.5:
            lat2, lon2, hist2, curr2 = lat1, lon1, hist1, curr1
            label = 0  # same location (no change)
        else:
            lat2, lon2, hist2, curr2 = random.choice(self.tiles)
            label = 1  # assumed change

        # Apply transform if available
        if self.transform:
            augmented = self.transform(image=hist1, image0=curr2)
            hist = augmented["image"].float()
            curr = augmented["image0"].float()
        else:
            hist = torch.tensor(hist1).permute(2, 0, 1).float()
            curr = torch.tensor(curr2).permute(2, 0, 1).float()

        return hist, curr, torch.tensor(label, dtype=torch.float)


# ---------------------- Main Training Script ----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate location grids (SF, NY, Seattle)
    locations = []
    for i in range(20):
        for j in range(20):
            lat = 37.70 + i * 0.005
            lon = -122.52 + j * 0.005
            locations.append((lat, lon, 0))

    for i in range(15):
        for j in range(15):
            lat = 40.70 + i * 0.005
            lon = -74.02 + j * 0.005
            locations.append((lat, lon, 0))

    for i in range(15):
        for j in range(15):
            lat = 47.60 + i * 0.005
            lon = -122.35 + j * 0.005
            locations.append((lat, lon, 0))

    print(f"Generated {len(locations)} training points")

    # Prefetch tiles with a progress bar
    print("Fetching tiles... (this may take several minutes)")
    fetched_tiles = []
    for lat, lon, _ in tqdm(locations, desc="Downloading tiles", unit="tile"):
        hist = fetch_gee_tile(lat, lon)
        curr = fetch_google_tile(lat, lon)

        # Handle missing tiles
        if hist is None:
            hist = np.zeros((256, 256, 3), dtype=np.uint8)
        if curr is None:
            curr = np.zeros((256, 256, 3), dtype=np.uint8)

        # Ensure HWC uint8
        if hist.dtype != np.uint8:
            hist = np.clip(hist * 255, 0, 255).astype(np.uint8)
        if curr.dtype != np.uint8:
            curr = np.clip(curr * 255, 0, 255).astype(np.uint8)

        fetched_tiles.append((lat, lon, hist, curr))

    print(f"Fetched {len(fetched_tiles)} total tiles successfully.")

    # Dataset and DataLoader
    train_transform = ChangeDetectionDataset(locations).transform
    train_dataset = ContrastiveDataset(fetched_tiles, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    # Model, loss, optimizer
    model = ResNet50Siamese(pretrained=True).to(device)
    # Train the classification head directly with logits
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------------------- Training Loop ----------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for hist, curr, label in progress_bar:
            hist = hist.to(device)
            curr = curr.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # Compute embeddings for both inputs
            out1 = model.forward_once(hist)
            out2 = model.forward_once(curr)
            # Classification head on absolute difference (logits, no sigmoid)
            diff = torch.abs(out1 - out2)
            logits = model.fc(diff).squeeze(1)
            loss = bce_criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * hist.size(0)
            progress_bar.set_postfix({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss = {epoch_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model_resnet_grid.pth")
    print("✅ Model saved to models/model_resnet_grid.pth")
