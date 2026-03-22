import sys
import argparse
from scraper import scrape_data
from parser import parse_all_data

def main():
    parser = argparse.ArgumentParser(description="Agricultural Data Scraper Automation")
    parser.add_argument(
        "--scrape", 
        action="store_true", 
        help="Run the Selenium web scraper to download data"
    )
    parser.add_argument(
        "--parse", 
        action="store_true", 
        help="Run the HTML parser to generate the final dataset"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run scraper in headless mode (invisible browser)"
    )
    
    args = parser.parse_args()
    
    if not args.scrape and not args.parse:
        print("⚠️ Please specify an action. Use --scrape, --parse, or both.")
        print("Example: python src/main.py --scrape --parse")
        sys.exit(1)
        
    if args.scrape:
        print("🚀 Starting Web Scraper...")
        scrape_data(headless=args.headless)
        
    if args.parse:
        print("🚀 Starting HTML Parser...")
        parse_all_data()
        
    print("✨ Operations completed successfully.")

if __name__ == "__main__":
    main()
