from torch.utils.data import Dataset
import random
from fetch_historical import fetch_gee_tile as fetch_landsat_tile
from fetch_current import fetch_google_tile


class ChangeDetectionDataset(Dataset):
    def __init__(self, locations, transform=None):
        """
        Args:
            locations (list of (lat, lon, label)): label=1 if change, 0 if no change
        """
        self.locations = locations
        self.transform = transform

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        lat, lon, label = self.locations[idx]
        hist = fetch_landsat_tile(lat, lon)
        curr = fetch_google_tile(lat, lon)

        if hist is None or curr is None:
            # fallback: pick a random other sample
            return self[random.randint(0, len(self.locations)-1)]

        # Resize to 256x256 just in case
        hist = cv2.resize(hist, (256,256))
        curr = cv2.resize(curr, (256,256))

        # To tensor
        hist_tensor = torch.tensor(hist.transpose(2,0,1), dtype=torch.float32)
        curr_tensor = torch.tensor(curr.transpose(2,0,1), dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return hist_tensor, curr_tensor, label_tensor
