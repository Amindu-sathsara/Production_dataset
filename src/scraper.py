import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
URL = "https://www.statistics.gov.lk/HIES/HIES2006_07Website/"
CROPS = ["Beans", "Carrot", "Tomato", "Cabbage", "Beetroot"]
RAW_DATA_DIR = "data/raw"


def create_driver(headless=False):
    """Creates an optimized Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def select_all_districts(driver, wait):
    """
    Tries to click the "Select All" button / checkbox for districts.
    If not found, attempts to click each district checkbox individually.
    """
    try:
        # Look for a "Select All" element (common patterns)
        select_all = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//input[@value='Select All']|//a[contains(text(),'Select All')]"))
        )
        select_all.click()
        logging.info("✅ Selected all districts using 'Select All' control.")
    except Exception:
        logging.warning("⚠️ 'Select All' control not found – falling back to clicking each district checkbox.")
        # Find all checkboxes that look like districts (e.g., id starts with "chkDistrict")
        district_checkboxes = driver.find_elements(By.XPATH, "//input[@type='checkbox'][contains(@id,'chkDistrict')]")
        if not district_checkboxes:
            raise Exception("No district checkboxes found.")
        for cb in district_checkboxes:
            if not cb.is_selected():
                cb.click()
        logging.info(f"✅ Selected {len(district_checkboxes)} districts individually.")


def scrape_data(headless=True):
    """
    Scrapes each crop once for the full available year range.
    Saves the resulting HTML in data/raw/ as <Crop>.html
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    driver = create_driver(headless)
    wait = WebDriverWait(driver, 20)

    total_tasks = len(CROPS)
    logging.info(f"Starting scraping for {total_tasks} crops...")

    try:
        with tqdm(total=total_tasks, desc="Scraping Progress") as pbar:
            for crop in CROPS:
                file_path = os.path.join(RAW_DATA_DIR, f"{crop}.html")

                # Skip if already cached
                if os.path.exists(file_path):
                    logging.info(f"⏭️ Skipping {crop} (already exists)")
                    pbar.update(1)
                    continue

                logging.info(f"🔄 Processing crop: {crop}")

                try:
                    driver.get(URL)

                    # 1. Wait for the page to be ready
                    wait.until(EC.presence_of_element_located((By.ID, "optCategory")))

                    # 2. Select all districts
                    select_all_districts(driver, wait)

                    # 3. Select category "Up Country Vegetable"
                    category_select = Select(wait.until(EC.presence_of_element_located((By.ID, "optCategory"))))
                    # Ensure the dropdown has options
                    wait.until(lambda d: len(category_select.options) > 1)

                    target_category = None
                    for opt in category_select.options:
                        text = opt.text.strip().lower()
                        if text == "up country vegetable" or ("up country" in text and "veg" in text):
                            target_category = opt.text
                            break
                    if not target_category:
                        raise Exception("Category 'Up Country Vegetable' not found.")
                    category_select.select_by_visible_text(target_category)
                    logging.info(f"✅ Selected category: {target_category}")

                    # 4. Check the extent and production checkboxes (if they exist)
                    try:
                        extent_chk = driver.find_element(By.ID, "chkExtent")
                        if not extent_chk.is_selected():
                            extent_chk.click()
                        prod_chk = driver.find_element(By.ID, "chkProduction")
                        if not prod_chk.is_selected():
                            prod_chk.click()
                    except:
                        pass

                    # Wait for crop dropdown to be populated
                    time.sleep(1.5)
                    crop_dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "optCrop"))))
                    wait.until(lambda d: len(crop_dropdown.options) > 1)

                    # Select the crop
                    crop_options = [opt.text.strip() for opt in crop_dropdown.options]
                    if crop not in crop_options:
                        logging.warning(f"⚠️ Crop '{crop}' not available. Options: {crop_options}")
                        pbar.update(1)
                        continue
                    crop_dropdown.select_by_visible_text(crop)
                    logging.info(f"✅ Selected crop: {crop}")

                    # 5. Select year range: from the first available year to the last
                    # Wait for year dropdowns
                    wait.until(lambda d: len(Select(d.find_element(By.ID, "optFromYear")).options) > 1)
                    from_select = Select(driver.find_element(By.ID, "optFromYear"))
                    to_select = Select(driver.find_element(By.ID, "optToYear"))

                    # Get the first and last available year texts
                    from_options = [opt.text.strip() for opt in from_select.options]
                    to_options = [opt.text.strip() for opt in to_select.options]

                    # Usually the first option is "2001" and last is "2025" or similar
                    from_year_text = from_options[1] if len(from_options) > 1 else from_options[0]
                    to_year_text = to_options[-1]

                    from_select.select_by_visible_text(from_year_text)
                    to_select.select_by_visible_text(to_year_text)
                    logging.info(f"✅ Selected year range: {from_year_text} – {to_year_text}")

                    # 6. Click "Get Table"
                    driver.find_element(By.XPATH, "//input[@value='Get Table']").click()

                    # 7. Wait for the table to appear
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                    time.sleep(1)  # allow any final rendering

                    # 8. Save HTML source
                    html_source = driver.page_source
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(html_source)

                    logging.info(f"✅ Successfully saved {file_path}")

                except Exception as e:
                    logging.error(f"❌ Failed processing {crop}: {e}")
                    # Optionally restart browser if session died
                    if "invalid session id" in str(e).lower() or "no such window" in str(e).lower():
                        logging.error("🚨 Session crashed! Restarting browser...")
                        driver.quit()
                        driver = create_driver(headless)
                        wait = WebDriverWait(driver, 20)

                pbar.update(1)

    finally:
        try:
            driver.quit()
        except:
            pass
        logging.info("🛑 Scraping finished.")


if __name__ == "__main__":
    scrape_data(headless=False)  # Set to True for headless run