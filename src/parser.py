import os
import glob
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def parse_html_to_df(file_path):
    """
    Parses a single HTML file into the required Dataframe format.
    Required Columns:
    - year
    - Season
    - Location_district
    - Crop_nam
    - harvested
    - Productio
    - yields =production/Extent
    """
    try:
        # Extract metadata from filename
        filename = os.path.basename(file_path).replace(".html", "")
        year, crop = filename.split("_")
        
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        
        if table is None:
            logging.warning(f"⚠️ No table found in {file_path}")
            return None
            
        dfs = pd.read_html(str(table))
        if not dfs:
            return None
            
        df = dfs[0]
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(i) for i in col if i]) for col in df.columns]
            
        # The first few rows usually contain header artifacts, but read_html handles most.
        # Let's clean the column names based on the structure.
        # Structure is usually: District, Extent(Yala, Maha, Total), Production(Yala, Maha, Total)
        # Assuming the standard 7 columns are present:
        if len(df.columns) < 7:
            logging.warning(f"⚠️ Incomplete columns in {file_path}: {list(df.columns)}")
            return None
            
        df.columns = [
            "District",
            "Extent_Yala", "Extent_Maha", "Extent_Total",
            "Production_Yala", "Production_Maha", "Production_Total"
        ]
        
        # Drop summary rows or exact NaNs
        df = df.dropna(subset=["District"])
        df = df[~df["District"].str.contains("Total", case=False, na=False)]
        
        # Melt to rows: Yala and Maha separately
        
        # Yala slice
        df_yala = df[["District", "Extent_Yala", "Production_Yala"]].copy()
        df_yala = df_yala.rename(columns={"Extent_Yala": "harvested", "Production_Yala": "Productio"})
        df_yala["Season"] = "Yala"
        
        # Maha slice
        df_maha = df[["District", "Extent_Maha", "Production_Maha"]].copy()
        df_maha = df_maha.rename(columns={"Extent_Maha": "harvested", "Production_Maha": "Productio"})
        df_maha["Season"] = "Maha"
        
        # Combine
        result_df = pd.concat([df_yala, df_maha], ignore_index=True)
        
        # Add metadata
        result_df["year"] = int(year)
        result_df["Crop_nam"] = crop
        result_df = result_df.rename(columns={"District": "Location_district"})
        
        # Convert to numeric
        result_df["harvested"] = pd.to_numeric(result_df["harvested"], errors="coerce").fillna(0)
        result_df["Productio"] = pd.to_numeric(result_df["Productio"], errors="coerce").fillna(0)
        
        # Calculate yield
        result_df["yields =production/Extent"] = np.where(
            result_df["harvested"] > 0, 
            result_df["Productio"] / result_df["harvested"], 
            0
        )
        
        # Reorder columns to target output
        final_cols = [
            "year", "Season", "Location_district", "Crop_nam", 
            "harvested", "Productio", "yields =production/Extent"
        ]
        return result_df[final_cols]
        
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
        
        # Sort by year, season, location for cleanliness
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
