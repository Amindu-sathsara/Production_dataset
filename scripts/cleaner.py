import pandas as pd

df = pd.read_csv("../data/raw/raw_data.csv")

# Rename columns
df.columns = [
    "District",
    "Extent_Yala", "Extent_Maha", "Extent_Total",
    "Production_Yala", "Production_Maha", "Production_Total",
    "Year", "Crop"
]

# Remove empty rows
df = df.dropna(subset=["District"])

# Convert numeric columns
cols = df.columns[1:7]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Fill missing values
df = df.fillna(0)

# Save clean dataset
df.to_csv("../output/final_dataset.csv", index=False)

print("✅ Clean dataset ready")