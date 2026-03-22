import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

URL = "https://www.statistics.gov.lk/HIES/HIES2006_07Website/"
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

driver.get(URL)

print("Waiting for category...")
cat_elem = wait.until(EC.presence_of_element_located((By.ID, "optCategory")))

# Wait for JS to populate
time.sleep(3)

select = Select(cat_elem)
opts = [opt.text for opt in select.options]
print("AVAILABLE CATEGORIES:")
for o in opts:
    print(repr(o))

driver.quit()
