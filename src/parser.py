import os
import glob
import re
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def parse_html_to_df(file_path):
    """
    Parses the multi‑year HTML table into a tidy DataFrame.
    Expected columns: year, Season, Location_district, Crop_nam, harvested, Productio, yield
    """
    try:
        # Extract crop name from filename
        filename = os.path.basename(file_path)
        crop = filename.replace(".html", "")

        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if table is None:
            logging.warning(f"⚠️ No table found in {file_path}")
            return None

        # ---- 1. Parse header rows to build column metadata ----
        rows = table.find_all("tr")
        if len(rows) < 3:
            logging.warning(f"⚠️ Table has fewer than 3 header rows in {file_path}")
            return None

        # First row: contains years (with colspan=6 each)
        year_row = rows[0]
        year_spans = []
        for td in year_row.find_all("td"):
            colspan = int(td.get("colspan", 1))
            # Extract year from text (e.g., "2001")
            text = td.get_text(strip=True)
            year = None
            match = re.search(r"\b(19|20)\d{2}\b", text)
            if match:
                year = int(match.group())
            year_spans.append((year, colspan))

        # Build a list of column groups: each group is [year, type, season]
        # type = "harvested" or "Productio", season = "Yala" or "Maha" or "Total"
        # We'll ignore "Total" because the parser expects only Yala and Maha
        col_metadata = []  # list of (year, season, data_type) for each column (except district col)

        # Second row: tells us if a group of 3 columns is "Extent" or "Production"
        type_row = rows[1]
        type_tds = type_row.find_all("td")
        # Third row: tells us "Yala", "Maha", "Total"
        season_row = rows[2]
        season_tds = season_row.find_all("td")

        # We need to match the groups from the first row to the type and season rows
        # The first row groups of 6 columns (Extent Yala, Extent Maha, Extent Total, Production Yala, Production Maha, Production Total)
        # The second row groups of 3 columns: "Extent" and "Production" repeated
        # The third row repeats "Yala", "Maha", "Total" for each group

        # Instead of complex alignment, we can just walk through all columns (starting from index 1 because first column is district)
        # and assign metadata based on the pattern.

        # First, determine how many data columns exist (excluding the district column)
        # The district column is the first td in each data row, but we can count from the header
        # The header may have rowspan on the first column; but we can count columns by the number of tds in the third row (since it has no rowspan)
        # Actually the third row's tds count should equal the total number of columns (including district).
        # But we'll just iterate over the columns in the first data row later.

        # A simpler approach: extract the actual data rows, and for each row, we can parse the district and then for each year/season we need to know where to look.
        # We can use the year row to know how many columns belong to each year, then within each year, we know the order: Extent Yala, Extent Maha, Extent Total, Production Yala, Production Maha, Production Total.
        # So we can just loop over years and for each year, pick the 6 columns.

        # First, collect the years from the first row along with the starting column index for each year.
        col_idx = 1  # after the district column
        year_blocks = []  # list of (year, start_idx)
        for year, span in year_spans:
            year_blocks.append((year, col_idx))
            col_idx += span

        # Now iterate over data rows (starting from row 3, index 3)
        data_rows = rows[3:]
        all_records = []

        for row in data_rows:
            cells = row.find_all("td")
            if not cells:
                continue
            # First cell is district name
            district = cells[0].get_text(strip=True)
            if district == "" or district.lower() == "total":
                continue

            # For each year block, extract the 6 values
            for year, start in year_blocks:
                # Check if we have enough cells
                if start + 5 >= len(cells):
                    logging.warning(f"Not enough columns for year {year} in district {district}")
                    continue

                # Extent values
                ext_yala = cells[start].get_text(strip=True) if start < len(cells) else "0"
                ext_maha = cells[start+1].get_text(strip=True) if start+1 < len(cells) else "0"
                # ext_total = cells[start+2]  # we ignore total

                # Production values
                prod_yala = cells[start+3].get_text(strip=True) if start+3 < len(cells) else "0"
                prod_maha = cells[start+4].get_text(strip=True) if start+4 < len(cells) else "0"
                # prod_total = cells[start+5]

                # Helper to clean numeric strings (remove commas, replace &nbsp; with 0)
                def clean_num(val):
                    if not val or val == "&nbsp;" or val == "":
                        return 0.0
                    val = val.replace(",", "").replace("&nbsp;", "").strip()
                    try:
                        return float(val)
                    except:
                        return 0.0

                # Add Yala season record
                rec_yala = {
                    "year": year,
                    "Season": "Yala",
                    "Location_district": district,
                    "Crop_nam": crop,
                    "harvested": clean_num(ext_yala),
                    "Productio": clean_num(prod_yala),
                }
                # Add Maha season record
                rec_maha = {
                    "year": year,
                    "Season": "Maha",
                    "Location_district": district,
                    "Crop_nam": crop,
                    "harvested": clean_num(ext_maha),
                    "Productio": clean_num(prod_maha),
                }
                all_records.extend([rec_yala, rec_maha])

        if not all_records:
            logging.warning(f"No data records extracted from {file_path}")
            return None

        df = pd.DataFrame(all_records)
        # Calculate yield = production / harvested (handle division by zero)
        df["yields =production/Extent"] = np.where(
            df["harvested"] > 0,
            df["Productio"] / df["harvested"],
            0.0
        )
        return df

    except Exception as e:
        logging.error(f"❌ Error parsing {file_path}: {e}")
        return None


def parse_all_data():
    """
    Parses all cached HTML files and outputs the final dataset.
    """
    logging.info("Starting parsing process...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    html_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.html"))
    if not html_files:
        logging.warning("⚠️ No HTML files found in data/raw/. Please run the scraper first.")
        print("No HTML files found to parse!")
        return

    all_data = []
    for file_path in tqdm(html_files, desc="Parsing HTML Files"):
        df = parse_html_to_df(file_path)
        if df is not None and not df.empty:
            all_data.append(df)

    if all_data:
        final_dataset = pd.concat(all_data, ignore_index=True)
        final_dataset = final_dataset.sort_values(by=["year", "Season", "Location_district"])

        output_path_csv = os.path.join(PROCESSED_DATA_DIR, "agricultural_data.csv")
        output_path_excel = os.path.join(PROCESSED_DATA_DIR, "agricultural_data.xlsx")

        final_dataset.to_csv(output_path_csv, index=False)
        final_dataset.to_excel(output_path_excel, index=False)

        logging.info(f"✅ Successfully created final datasets at {PROCESSED_DATA_DIR}")
        print(f"✅ Parsing complete! Output saved to: {output_path_excel}")
    else:
        logging.error("❌ Parsing resulted in no data.")
        print("❌ No data was extracted.")


if __name__ == "__main__":
    parse_all_data()