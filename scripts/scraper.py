from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from tqdm import tqdm
import time

from parser import parse_table


# 🔥 Create driver function
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Chrome(options=options)


# Initialize
driver = create_driver()
wait = WebDriverWait(driver, 20)

URL = "https://www.statistics.gov.lk/HIES/HIES2006_07Website/"

years = list(range(2001, 2026))
crops = ["Beans", "Carrot", "Tomato", "Cabbage", "Beetroot"]

all_data = []
iteration = 0


# 🔄 MAIN LOOP
for year in tqdm(years):
    for crop in crops:
        try:
            iteration += 1

            # 🔥 Restart browser periodically
            if iteration % 20 == 0:
                print("♻️ Restarting browser to avoid crash...")
                try:
                    driver.quit()
                except:
                    pass

                driver = create_driver()
                wait = WebDriverWait(driver, 20)

            print(f"🔄 Processing: {year} - {crop}")

            # Load page
            driver.get(URL)

            # Wait until form loads
            wait.until(EC.presence_of_element_located((By.NAME, "category")))

            # ✅ Select category
            Select(driver.find_element(By.NAME, "category")).select_by_visible_text("Up Country Veg")

            # 🔥 IMPORTANT: wait for crop dropdown to update
            time.sleep(1)

            # ✅ SAFE CROP SELECTION
            crop_dropdown = wait.until(
                EC.presence_of_element_located((By.NAME, "crop"))
            )

            # Wait until options are loaded
            wait.until(lambda d: len(Select(crop_dropdown).options) > 1)

            crop_options = [opt.text.strip() for opt in Select(crop_dropdown).options]

            print("Available crops:", crop_options)

            if crop not in crop_options:
                print(f"⚠️ Skipping {crop} for {year} (not available)")
                continue

            Select(crop_dropdown).select_by_visible_text(crop)

            # Select year range
            Select(driver.find_element(By.NAME, "from")).select_by_visible_text(str(year))
            Select(driver.find_element(By.NAME, "to")).select_by_visible_text(str(year))

            # Click button
            driver.find_element(By.XPATH, "//input[@value='Get Table']").click()

            # Wait for table
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

            time.sleep(1)

            # Extract HTML
            html = driver.page_source

            # Parse
            df = parse_table(html, year, crop)

            if df is not None:
                all_data.append(df)

        except Exception as e:
            print(f"❌ FAILED: {year}, {crop}")
            print("Reason:", str(e))

            # 🔥 Recover session crash
            if "invalid session id" in str(e).lower():
                print("🚨 Session crashed! Restarting browser...")

                try:
                    driver.quit()
                except:
                    pass

                driver = create_driver()
                wait = WebDriverWait(driver, 20)

            continue


# 🧹 Cleanup
try:
    driver.quit()
except:
    pass


# 💾 Save data
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("../data/raw/raw_data.csv", index=False)
    print("✅ SUCCESS: Data scraping completed!")
else:
    print("⚠️ WARNING: No data collected!")