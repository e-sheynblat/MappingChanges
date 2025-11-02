import os
import requests
import cv2 # type: ignore
import numpy as np # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_API_KEY")

def fetch_google_tile(lat, lon, zoom=16, size=256):
    """Fetch a satellite tile from Google Maps in memory."""
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("Missing GOOGLE_MAPS_API_KEY in environment")
    
    url = (f"https://maps.googleapis.com/maps/api/staticmap?"
           f"center={lat},{lon}&zoom={zoom}&size={size}x{size}"
           f"&maptype=satellite&key={GOOGLE_MAPS_API_KEY}")

    resp = requests.get(url)
    if resp.status_code != 200:
        print("Google Maps API error:", resp.status_code)
        return None

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0
