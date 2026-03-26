import pandas as pd
import argparse
import sys

# Standard month name mapping
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

def expand_to_monthly(df, season_col='Season', 
                      yala_months=(4,5,6,7,8,9),    # Apr to Sep
                      maha_months=(10,11,12,1,2,3)): # Oct to Mar
    """
    Expand seasonal rows into monthly rows.
    """
    season_map = {
        'Yala': (yala_months, len(yala_months)),
        'Maha': (maha_months, len(maha_months))
    }
    
    expanded_rows = []
    for _, row in df.iterrows():
        season = row[season_col]
        if season not in season_map:
            print(f"Warning: unknown season '{season}' in row, skipping.")
            continue
        
        months, n_months = season_map[season]
        harvested_total = row['harvested']
        production_total = row['Productio']   # <-- fixed column name
        
        if n_months == 0:
            continue
        
        harvested_monthly = harvested_total / n_months
        production_monthly = production_total / n_months
        yield_monthly = production_monthly / harvested_monthly if harvested_monthly != 0 else 0
        
        for month_num in months:
            expanded_rows.append({
                'year': row['year'],
                '_month_num': month_num,                 # store number for sorting
                'Location_district': row['Location_district'],
                'Crop_nam': row['Crop_nam'],
                'harvested': harvested_monthly,
                'Productio': production_monthly,
                'yield': yield_monthly
            })
    
    result = pd.DataFrame(expanded_rows)
    # Sort by year and month number
    result = result.sort_values(by=['year', '_month_num']).reset_index(drop=True)
    # Convert month number to name
    result['month'] = result['_month_num'].map(MONTH_NAMES)
    # Drop temporary number column
    result = result.drop(columns=['_month_num'])
    # Reorder columns
    result = result[['year', 'month', 'Location_district', 'Crop_nam', 'harvested', 'Productio', 'yield']]
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert seasonal crop data to monthly granularity.")
    parser.add_argument('input', help='Path to input CSV or Excel file (must contain columns: year, Season, Location_district, Crop_nam, harvested, Productio)')
    parser.add_argument('--output', '-o', default='monthly_agricultural_data.csv',
                        help='Output file name (default: monthly_agricultural_data.csv)')
    parser.add_argument('--yala-months', nargs='+', type=int, default=[4,5,6,7,8,9],
                        help='Months for Yala (default: 4 5 6 7 8 9)')
    parser.add_argument('--maha-months', nargs='+', type=int, default=[10,11,12,1,2,3],
                        help='Months for Maha (default: 10 11 12 1 2 3)')
    args = parser.parse_args()
    
    # Read input file
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(args.input)
    else:
        print("Input file must be CSV or Excel.")
        sys.exit(1)
    
    # Check required columns – note the correct production column name
    required = ['year', 'Season', 'Location_district', 'Crop_nam', 'harvested', 'Productio']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        sys.exit(1)
    
    # Clean year column
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Expand
    monthly_df = expand_to_monthly(df, yala_months=tuple(args.yala_months), maha_months=tuple(args.maha_months))
    
    # Save output
    if args.output.endswith('.csv'):
        monthly_df.to_csv(args.output, index=False)
    elif args.output.endswith(('.xlsx', '.xls')):
        monthly_df.to_excel(args.output, index=False)
    else:
        monthly_df.to_csv(args.output, index=False)
    
    print(f"✅ Monthly data saved to {args.output}")
    print(f"Total rows: {len(monthly_df)}")
    print("Columns:", list(monthly_df.columns))

if __name__ == '__main__':
    main()