import os
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_API_KEY")

def fetch_google_tile(lat, lon, zoom=16):
    """Fetch current satellite tile from Google Maps in memory."""
    base = "https://maps.googleapis.com/maps/api/staticmap?"
    url = f"{base}center={lat},{lon}&zoom={zoom}&size=640x640&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Google Maps API error:", response.status_code)
        return None

    img_array = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
