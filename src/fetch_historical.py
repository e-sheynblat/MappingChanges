import ee # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import requests
from io import BytesIO

# Authenticate & initialize GEE
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize(project='map-change-detection')

def fetch_gee_tile(lat, lon, start_date="2025-03-01", end_date="2025-09-30", size=256):
    """Fetch Sentinel-2 tile from Google Earth Engine."""
    delta = 0.005
    region = ee.Geometry.Rectangle([lon - delta, lat - delta, lon + delta, lat + delta])

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .sort('system:time_start', False))

    if collection.size().getInfo() == 0:
        print(f"[WARNING] No images for {lat},{lon}")
        return None

    image = collection.first()
    vis_params = {'bands':['B4','B3','B2'], 'min':0, 'max':3000}
    url = image.getThumbURL({'region':region, 'dimensions':size, 'format':'png',
                             'min':vis_params['min'], 'max':vis_params['max'],
                             'bands':vis_params['bands']})

    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0
        return arr
    except Exception as e:
        print(f"[ERROR] Failed to fetch GEE tile: {e}")
        return None
