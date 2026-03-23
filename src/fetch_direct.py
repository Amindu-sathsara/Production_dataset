import requests
import os

# URLs for each crop (as you provided)
CROP_URLS = {
    "Beans": "https://www.statistics.gov.lk/HIES/HIES2006_07Website/HighlandCrops.asp?getcode=Table&dC=25&dL=11121321222331323341424344455152536162717281829192&P=1&E=1&C=35&F=10&T=34",
    "Cabbage": "https://www.statistics.gov.lk/HIES/HIES2006_07Website/HighlandCrops.asp?getcode=Table&dC=25&dL=11121321222331323341424344455152536162717281829192&P=1&E=1&C=30&F=10&T=34",
    "Beetroot": "https://www.statistics.gov.lk/HIES/HIES2006_07Website/HighlandCrops.asp?getcode=Table&dC=25&dL=11121321222331323341424344455152536162717281829192&P=1&E=1&C=33&F=10&T=34",
    "Carrot": "https://www.statistics.gov.lk/HIES/HIES2006_07Website/HighlandCrops.asp?getcode=Table&dC=25&dL=11121321222331323341424344455152536162717281829192&P=1&E=1&C=31&F=10&T=34",
    "Tomatoes": "https://www.statistics.gov.lk/HIES/HIES2006_07Website/HighlandCrops.asp?getcode=Table&dC=25&dL=11121321222331323341424344455152536162717281829192&P=1&E=1&C=28&F=10&T=34"
}

# Create the raw data directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

for crop, url in CROP_URLS.items():
    print(f"Downloading {crop}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Save the HTML content to data/raw/<Crop>.html
            file_path = os.path.join("data/raw", f"{crop}.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"  ✅ Saved {crop}.html")
        else:
            print(f"  ❌ Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error downloading {crop}: {e}")

print("All downloads attempted. Check data/raw/ for the files.")