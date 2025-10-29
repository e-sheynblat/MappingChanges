from fetch_historical import fetch_landsat_tile
from fetch_current import fetch_google_tile
import torch
import cv2

def process_tile_pair(lat, lon, model):
    """Fetch historical & current tiles in memory, run Siamese model, return prediction JSON."""
    hist = fetch_landsat_tile(lat, lon)
    curr = fetch_google_tile(lat, lon)

    if hist is None or curr is None:
        return None

    # Resize to 256x256
    hist = cv2.resize(hist, (256,256))
    curr = cv2.resize(curr, (256,256))

    # Convert to PyTorch tensors
    hist_tensor = torch.tensor(hist.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)
    curr_tensor = torch.tensor(curr.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = model(hist_tensor, curr_tensor).item()

    return {"lat": lat, "lon": lon, "change": score}
