import torch
import json
from siamese_model import ResNet50Siamese
from fetch_historical import fetch_gee_tile
from fetch_current import fetch_google_tile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = ResNet50Siamese(pretrained=False).to(device)
model.load_state_dict(torch.load("models/model_resnet_grid.pth", map_location=device))
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
predictions_for_heatmap = []

# Threshold for classification
THRESHOLD = 0.5

for lat, lon, label in test_set:
    hist = fetch_gee_tile(lat, lon)
    curr = fetch_google_tile(lat, lon)
    if hist is None or curr is None:
        continue
    with torch.no_grad():
        hist_t = torch.tensor(hist.transpose(2,0,1)).unsqueeze(0).to(device)
        curr_t = torch.tensor(curr.transpose(2,0,1)).unsqueeze(0).to(device)
        prob = model(hist_t, curr_t).item()
    
    y_true.append(label)
    y_pred.append(1 if prob >= THRESHOLD else 0)
    
    predictions_for_heatmap.append({
        "lat": lat,
        "lon": lon,
        "prob": prob
    })

# Compute metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")

# Save predictions for D3.js heatmap
with open("predictions.json", "w") as f:
    json.dump(predictions_for_heatmap, f, indent=2)

print("Saved predictions.json for visualization")
