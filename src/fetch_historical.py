import ee
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Authenticate & initialize GEE
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize(project='map-change-detection')

def fetch_gee_tile(lat, lon, start_date="2022-01-01", end_date="2022-12-31", size=256):
    """
    Fetch a Sentinel-2 image tile from Google Earth Engine for a given lat/lon and date range.
    Returns a 3-channel RGB image (numpy array) in memory, resized to `size x size`.

    Args:
        lat (float): Latitude of center point.
        lon (float): Longitude of center point.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        size (int): Output image width/height in pixels.

    Returns:
        np.ndarray: RGB image as (size, size, 3) float32 array, normalized to [0,1].
    """
    delta = 0.005  # ~0.5km
    region = ee.Geometry.Rectangle([lon - delta, lat - delta, lon + delta, lat + delta])

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .sort('system:time_start', False))

    if collection.size().getInfo() == 0:
        print("[WARNING] No images found for location/date.")
        return None

    image = collection.first()

    # Visualization parameters: scale 0-3000 to convert to 8-bit RGB
    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000
    }

    url = image.getThumbURL({'region': region, 'dimensions': size, 'format': 'png', 'min': vis_params['min'], 'max': vis_params['max'], 'bands': vis_params['bands']})

    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((size, size))
        img_array = np.array(img).astype(np.float32) / 255.0
    except Exception as e:
        print(f"[ERROR] Failed to fetch image from GEE: {e}")
        return None

    print(f"[INFO] Fetched GEE tile at lat={lat}, lon={lon}, shape={img_array.shape}")
    return img_array
