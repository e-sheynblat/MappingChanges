from src.fetch_historical import fetch_gee_tile

tile = fetch_gee_tile(37.7749, -122.4194)

if tile is not None:
    print("Fetched historical tile successfully! Shape:", tile.shape)
else:
    print("Failed to fetch historical tile.")
