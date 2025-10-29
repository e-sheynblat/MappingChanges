from src.fetch_current import fetch_google_tile

tile = fetch_google_tile(37.7749, -122.4194)
if tile is not None:
    print("Fetched current tile successfully! Shape:", tile.shape)
else:
    print("Failed to fetch current tile.")
