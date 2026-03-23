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
YEARS = list(range(2001, 2026))
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

def scrape_data(headless=True):
    """
    Scrapes the agricultural datasets and caches the HTML locally
    in data/raw/ to ensure network resilience and speed.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    driver = create_driver(headless)
    wait = WebDriverWait(driver, 20)
    
    iteration = 0
    total_tasks = len(YEARS) * len(CROPS)
    
    logging.info(f"Starting scraping process for {total_tasks} Year-Crop combinations...")
    
    try:
        with tqdm(total_tasks, desc="Scraping Progress") as pbar:
            for year in YEARS:
                for crop in CROPS:
                    iteration += 1
                    file_path = os.path.join(RAW_DATA_DIR, f"{year}_{crop}.html")
                    
                    # 1️⃣ Skip if we already downloaded it (Caching!)
                    if os.path.exists(file_path):
                        logging.info(f"⏭️ Skipping {year} - {crop} (Already exists)")
                        pbar.update(1)
                        continue
                    
                    # ♻️ Restart browser periodically to prevent memory leaks / crashes
                    if iteration % 20 == 0:
                        logging.info("♻️ Restarting browser to release memory...")
                        try:
                            driver.quit()
                        except:
                            pass
                        driver = create_driver(headless)
                        wait = WebDriverWait(driver, 20)
                        
                    logging.info(f"🔄 Processing: {year} - {crop}")
                    
                    try:
                        driver.get(URL)

                        # Wait for category dropdown to be present
                        wait.until(
                            EC.presence_of_element_located((By.ID, "optCategory"))
                        )

                        # Extra wait until the category <select> actually has options
                        def _category_has_options(d):
                            try:
                                elem = d.find_element(By.ID, "optCategory")
                                return len(Select(elem).options) > 1
                            except Exception:
                                return False

                        wait.until(_category_has_options)

                        category_element = driver.find_element(By.ID, "optCategory")
                        category_select = Select(category_element)

                        # Read all category option texts for logging & matching
                        category_options = [opt.text.strip() for opt in category_select.options]
                        logging.info(f"Available categories: {category_options}")

                        # Prefer exact 'Up Country Vegetable' if present
                        target_category_text = None
                        for text in category_options:
                            if text.strip().lower() == "up country vegetable":
                                target_category_text = text
                                break

                        # Fallback: any category that contains both 'Up Country' and 'Veg'
                        if not target_category_text:
                            for text in category_options:
                                lower = text.lower()
                                if "up country" in lower and "veg" in lower:
                                    target_category_text = text
                                    break

                        if not target_category_text:
                            logging.error(
                                f"Category 'Up Country Vegetable' not found. Available options: {category_options}"
                            )
                            pbar.update(1)
                            continue

                        category_select.select_by_visible_text(target_category_text)
                        
                        # Check Extent and Production if need be
                        try:
                            extent_chk = driver.find_element(By.ID, "chkExtent")
                            if not extent_chk.is_selected(): extent_chk.click()
                            prod_chk = driver.find_element(By.ID, "chkProduction")
                            if not prod_chk.is_selected(): prod_chk.click()
                        except:
                            pass

                        # Wait for Javascript to populate crop dropdown
                        time.sleep(1.5)
                        
                        crop_dropdown = wait.until(EC.presence_of_element_located((By.ID, "optCrop")))
                        wait.until(lambda d: len(Select(crop_dropdown).options) > 1)
                        
                        crop_options = [opt.text.strip() for opt in Select(crop_dropdown).options]
                        
                        if crop not in crop_options:
                            logging.warning(f"⚠️ Crop {crop} not found for {year}")
                            pbar.update(1)
                            continue
                            
                        Select(crop_dropdown).select_by_visible_text(crop)

                        # --- Robust year range selection ---

                        # Wait until the From-year dropdown has real options (JS finished)
                        def _year_has_options(d):
                            try:
                                sel = Select(d.find_element(By.ID, "optFromYear"))
                                return len(sel.options) > 1
                            except Exception:
                                return False

                        wait.until(_year_has_options)

                        from_select = Select(driver.find_element(By.ID, "optFromYear"))
                        to_select = Select(driver.find_element(By.ID, "optToYear"))

                        from_options = [opt.text.strip() for opt in from_select.options]
                        to_options = [opt.text.strip() for opt in to_select.options]

                        # Log available years (for debugging)
                        logging.info(f"From-year options: {from_options}")
                        logging.info(f"To-year options: {to_options}")

                        year_str = str(year)

                        def _find_year_text(options_list, y):
                            for text in options_list:
                                if y in text:
                                    return text
                            return None

                        from_text = _find_year_text(from_options, year_str)
                        to_text = _find_year_text(to_options, year_str)

                        if not from_text or not to_text:
                            logging.error(
                                f"❌ Year {year} not found in dropdowns. From options: {from_options}, To options: {to_options}"
                            )
                            pbar.update(1)
                            continue

                        from_select.select_by_visible_text(from_text)
                        to_select.select_by_visible_text(to_text)
                        
                        # Click Get Table
                        driver.find_element(By.XPATH, "//input[@value='Get Table']").click()
                        
                        # Wait for the table to load
                        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                        time.sleep(1) # Let all elements settle
                        
                        # 2️⃣ Save HTML source (Caching)
                        html_source = driver.page_source
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(html_source)
                            
                        logging.info(f"✅ Successfully downloaded {year} - {crop}")
                        
                    except Exception as e:
                        logging.error(f"❌ Failed processing {year} - {crop}: {e}")
                        # Browser crash recovery
                        if "invalid session id" in str(e).lower() or "no such window" in str(e).lower():
                            logging.error("🚨 Session crashed! Restarting immediately...")
                            try:
                                driver.quit()
                            except:
                                pass
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
    scrape_data(headless=False) # Keep visible for testing if run directly
