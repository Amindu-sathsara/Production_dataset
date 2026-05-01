# Agricultural Data Scraper

This repository contains a professional, robust web scraping architecture designed to extract agricultural yield data from the Sri Lankan government statistics website. 

The resulting dataset will exactly match the target analytical format with separate rows for `Yala` and `Maha` seasons, calculated yield metrics, and cleanly parsed location and crop information.

## Overview of the Architecture

1. **Scraper (`src/scraper.py`)**: Uses a headless Selenium browser to iterate through the years 2001-2026 and specific crops. It automatically navigates the dropdowns, waits for the data table to load, and then **caches the raw HTML** directly to disk (`data/raw/`). This ensures maximum resilience—if the scraper fails midway, it will resume from where it left off, and if the data extraction logic needs changing, we do not need to hit the website again.
2. **Parser (`src/parser.py`)**: Reads the cached HTML files using `BeautifulSoup` and converts them into Pandas DataFrames. It uses `pandas.melt` logic to dynamically separate the Yala and Maha columns into individual rows, cleans up column names, converts data types, and calculates the final `yields =production/Extent` column.
3. **Main Entrypoint (`src/main.py`)**: A command-line interface orchestrator tying it all together.

## Installation Requirements

Ensure you have Python installed, and then install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Navigate to the root directory `production/` in your terminal and use `src/main.py`:

**1. To Scrape the Data First (Downloads HTML code only):**
```bash
python src/main.py --scrape
```
*Note: Due to the amount of data, scraping multiple years will take some time. Check `logs/scraper.log` to track progress. If the network or website drops connection, simply re-run the command; it will skip already downloaded files!*

**2. To Parse the Cached Data into the Final Excel/CSV:**
```bash
python src/main.py --parse
```

**3. To Run Everything Back-to-Back:**
```bash
python src/main.py --scrape --parse
```

The final dataset will be generated efficiently and saved to `data/processed/agricultural_data.xlsx`.


preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── utils.py
│   └── app.py   

are the files that here I just used to do this tasks to create product prediction\



#To run and see the predictions in actions 

To use the ARIMA / SARIMA / Prophet results interactively by entering year, month, crop, and district, use the ts_forecasts.predict script.

From your src folder:

cd C:\Users\HP\Desktop\production\src
python -m ts_forecasts.predict 2027 June Cabbage Anuradhapura


python -m ts_forecasts.predict <year> <month> <crop> <district>

python -m ts_forecasts.predict 2026 June Cabbage Anuradhapura
python -m ts_forecasts.predict 2026 January Beans Badulla
python -m ts_forecasts.predict 2030 March Tomatoes Kandy