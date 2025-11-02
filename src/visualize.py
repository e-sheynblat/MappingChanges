import json
import torch # type: ignore
from siamese_model import SiameseNetwork
from fetch_historical import fetch_gee_tile
from fetch_current import fetch_google_tile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("models/model_v1.pth"))
model.eval()

# 1. Create grid of SF lat/lon
grid = [(37.70 + i*0.01, -122.52 + j*0.01) for i in range(8) for j in range(8)]

predictions = []
for lat, lon in grid:
    hist = fetch_gee_tile(lat, lon)
    curr = fetch_google_tile(lat, lon)
    if hist is None or curr is None:
        continue
    with torch.no_grad():
        hist_t = torch.tensor(hist.transpose(2,0,1)).unsqueeze(0).to(device)
        curr_t = torch.tensor(curr.transpose(2,0,1)).unsqueeze(0).to(device)
        p = model(hist_t, curr_t).item()
    predictions.append({"lat":lat, "lon":lon, "prob":p})

with open("predictions_sf.json", "w") as f:
    json.dump(predictions, f)
print("Saved predictions_sf.json")
