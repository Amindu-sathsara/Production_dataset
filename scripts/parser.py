from bs4 import BeautifulSoup
import pandas as pd

def parse_table(html, year, crop):
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table")

    if table is None:
        print(f"⚠️ No table found for {year} - {crop}")
        return None

    try:
        df = pd.read_html(str(table))[0]

        # Handle multi-level headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(i) for i in col if i])
                for col in df.columns
            ]

        # Rename columns (adjust if needed)
        df.columns = [
            "District",
            "Extent_Yala", "Extent_Maha", "Extent_Total",
            "Production_Yala", "Production_Maha", "Production_Total"
        ]

        # Clean data
        df = df[df["District"].notna()]

        numeric_cols = df.columns[1:]
        df[numeric_cols] = df[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        df = df.fillna(0)

        # Add metadata
        df["Year"] = year
        df["Crop"] = crop

        return df

    except Exception as e:
        print(f"❌ Parsing error: {year}, {crop}", e)
        return None