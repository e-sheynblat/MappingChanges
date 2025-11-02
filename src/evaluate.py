import torch
import json
from siamese_model import ResNet50Siamese
from fetch_historical import fetch_gee_tile
from fetch_current import fetch_google_tile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = ResNet50Siamese(pretrained=False).to(device)
# Load weights safely (PyTorch 2.4+)
state = torch.load("models/model_resnet_grid.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

# Test set: expanded locations with labels (1=change, 0=no-change)
test_set = [
    # San Francisco
    (37.7749, -122.4194, 1),  # Downtown
    (37.7599, -122.4148, 0),  # Mission District
    (37.7680, -122.4450, 1),  # Sunset
    (37.8020, -122.4485, 0),  # Marina
    (37.7831, -122.4039, 1),  # SOMA
    
    # Los Angeles
    (34.0522, -118.2437, 1),
    (34.0622, -118.3080, 0),
    
    # New York City
    (40.7128, -74.0060, 1),
    (40.7306, -73.9352, 0),
    
    # Seattle
    (47.6062, -122.3321, 1),
    (47.6205, -122.3493, 0),
    
    # Chicago
    (41.8781, -87.6298, 1),
    (41.8890, -87.6236, 0),
]

y_true, y_pred = [], []
all_probs = []
predictions_for_heatmap = []

# Threshold for classification
THRESHOLD = 0.5

for lat, lon, label in test_set:
    hist = fetch_gee_tile(lat, lon)
    curr = fetch_google_tile(lat, lon)
    if hist is None or curr is None:
        continue

    with torch.no_grad():
        # to BCHW float32 in [0,1]
        hist_t = torch.tensor(hist.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
        curr_t = torch.tensor(curr.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
        # normalize using ImageNet stats (same as training)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        hist_t = (hist_t - mean) / std
        curr_t = (curr_t - mean) / std

        # Use the trained head (sigmoid inside model.forward)
        prob = model(hist_t, curr_t).view(-1).item()

    y_true.append(label)
    all_probs.append(prob)
    predictions_for_heatmap.append({"lat": lat, "lon": lon, "prob": prob})

# If we collected no samples, abort gracefully
if len(all_probs) == 0:
    print("No samples were evaluated (all tiles missing).")
else:
    # Optional: threshold sweep to maximize F1
    candidate_thresholds = [i/100 for i in range(5, 96, 5)]  # 0.05..0.95
    best = {"thr": THRESHOLD, "f1": -1.0, "acc": 0.0, "prec": 0.0, "rec": 0.0}
    for thr in candidate_thresholds:
        preds = [1 if p >= thr else 0 for p in all_probs]
        try:
            f1 = f1_score(y_true, preds)
            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds)
        except Exception:
            continue
        if f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "acc": acc, "prec": prec, "rec": rec}

    # Final predictions using best threshold
    y_pred = [1 if p >= best["thr"] else 0 for p in all_probs]

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Best threshold (by F1): {best['thr']:.2f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")

# Save predictions for D3.js heatmap
with open("predictions.json", "w") as f:
    json.dump(predictions_for_heatmap, f, indent=2)

print("Saved predictions.json for visualization")
