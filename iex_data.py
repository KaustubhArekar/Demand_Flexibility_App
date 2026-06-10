import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def scrape_single_day(date,market):
    """
    Scrape IEX data for a single date
    
    Parameters:
    date: datetime object
    
    Returns:
    DataFrame with hourly data for that date
    """
    
    date_str = date.strftime('%d-%m-%Y')
    
    # Use same from and to date to get single day
    url = f"https://www.iexindia.com/market-data/{market}/market-snapshot?interval=ONE_FOURTH_HOUR&dp=SELECT_RANGE&showGraph=false&toDate={date_str}&fromDate={date_str}"
    
    print(f"Scraping: {date_str}")
    print(f"URL: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse all tables
        tables = pd.read_html(response.text)
        print(f"Found {len(tables)} tables on the page")
        
        # Find the main data table (usually the one with 24 hours of data)
        for i, table in enumerate(tables):
            print(f"Table {i}: Shape {table.shape}, Columns: {list(table.columns)[:3]}...")
            
            # Look for table with 'Time Block' column and multiple rows
            if 'Time Block' in str(table.columns) and len(table) > 20:
                # Add date column
                table['Date'] = date_str
                print(f"✅ Found data table with {len(table)} rows")
                return table
        
        print(f"⚠️ No suitable table found for {date_str}")
        return None
        
    except Exception as e:
        print(f"❌ Error scraping {date_str}: {e}")
        return None

def scrape_date_range(start_date, end_date,market):

    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    
    
    while current_date <= end_date:
        # Scrape single day
        daily_data = scrape_single_day(current_date,market)
        
        if daily_data is not None:
            all_data.append(daily_data)
            # print(f"   ✓ Successfully scraped {current_date.strftime('%d-%m-%Y')}")
        else:
            print(f"   ✗ Failed to scrape {current_date.strftime('%d-%m-%Y')}")
        
        # Move to next day
        current_date += timedelta(days=1)
        
        # Be respectful to the server
        time.sleep(1)
    
    # print(f"\n📊 Scraping complete!")
    # print(f"Successfully scraped {len(all_data)} out of {total_days} days")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None

