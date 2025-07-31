#!/usr/bin/env python
# coding: utf-8

"""
This script calculates and reports the total number of documents and characters
in the People's Daily corpus across all available years.

It scans the data directory for all yearly data files, processes them to
calculate statistics, and prints a summary.
"""

import pandas as pd
import os
import re
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# --- Configuration ---
# Set the root directory where the People's Daily data is stored.
# This path is assumed to be consistent with other scripts.
DATA_ROOT = Path.home() / "data" / "rmrb_1948-2024"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_available_years(data_dir: Path) -> list[int]:
    """
    Scans the data directory to find available years based on filename patterns.
    
    Args:
        data_dir: The path to the directory containing the data files.

    Returns:
        A sorted list of integers representing the years for which data is available.
    """
    available_years = []
    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        return available_years
        
    pattern = re.compile(r"人民日报(\d{4})年文本数据\.xlsx")
    filenames = os.listdir(data_dir)
    logging.info(f"Scanning {len(filenames)} files in '{data_dir}'...")

    for filename in filenames:
        match = pattern.match(filename)
        if match:
            available_years.append(int(match.group(1)))
            
    if available_years:
        available_years.sort()
        logging.info(f"Found data for {len(available_years)} years, from {available_years[0]} to {available_years[-1]}.")
    else:
        logging.warning(f"No data files matching the pattern '人民日报YYYY年文本数据.xlsx' were found.")
        
    return available_years

def load_year_data(year: int) -> pd.DataFrame:
    """
    Loads the Excel data file for a specific year into a pandas DataFrame.
    This function is adapted from the preprocessing script for consistency.

    Args:
        year: The year for which to load data.

    Returns:
        A pandas DataFrame containing the '标题' (title) and '文本内容' (content)
        columns. Returns an empty DataFrame if the file is not found or is invalid.
    """
    filename = f"人民日报{year}年文本数据.xlsx"
    file_path = DATA_ROOT / filename
    if file_path.exists():
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            if '标题' not in df.columns:
                df['标题'] = ""
            if '文本内容' not in df.columns:
                df['文本内容'] = ""
            return df[['标题', '文本内容']].fillna('')
        except Exception as e:
            logging.error(f"Error loading or processing {file_path}: {e}")
            return pd.DataFrame(columns=['标题', '文本内容'])
    else:
        logging.warning(f"File not found for year {year}: {file_path}")
        return pd.DataFrame(columns=['标题', '文本内容'])

def main():
    """
    Main function to orchestrate the corpus statistics calculation.
    """
    parser = argparse.ArgumentParser(
        description="Calculate statistics for the People's Daily corpus for a given year range."
    )
    parser.add_argument(
        '--start-year', 
        type=int, 
        help='The start year of the range (inclusive). Defaults to the earliest available year.'
    )
    parser.add_argument(
        '--end-year', 
        type=int, 
        help='The end year of the range (inclusive). Defaults to the latest available year.'
    )
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        logging.error(f"Data root directory not found at '{DATA_ROOT}'.")
        logging.error("Please ensure the DATA_ROOT path in the script is correct.")
        return

    all_available_years = find_available_years(DATA_ROOT)
    if not all_available_years:
        logging.warning("No data files found. Cannot calculate statistics.")
        return

    # Determine the range of years to process based on arguments
    start_year = args.start_year if args.start_year is not None else all_available_years[0]
    end_year = args.end_year if args.end_year is not None else all_available_years[-1]

    years_to_process = [year for year in all_available_years if start_year <= year <= end_year]

    if not years_to_process:
        logging.warning(f"No data within the specified year range [{start_year}-{end_year}] was found.")
        return

    total_documents = 0
    total_characters = 0

    for year in tqdm(years_to_process, desc="Calculating corpus statistics"):
        df_year = load_year_data(year)
        if df_year.empty:
            continue
        
        total_documents += len(df_year)
        
        # Calculate character count for both title and content
        char_count_in_year = (df_year['标题'].astype(str).str.len() + 
                              df_year['文本内容'].astype(str).str.len()).sum()
        total_characters += char_count_in_year

    # Format numbers for clear, human-readable output
    chars_in_hundred_millions = total_characters / 100_000_000
    
    # Determine effective start and end years from the actual processed data
    effective_start_year = years_to_process[0]
    effective_end_year = years_to_process[-1]

    logging.info("--- Corpus Statistics Summary ---")
    logging.info(f"Year range processed: {effective_start_year}-{effective_end_year}")
    logging.info(f"Total years processed: {len(years_to_process)}")
    logging.info(f"Total documents (articles): {total_documents:,}")
    logging.info(f"Total characters: {total_characters:,}")
    
    print("\n" + "="*50)
    print(f"人民日报语料库 ({effective_start_year}-{effective_end_year}) 总体规模统计".center(40))
    print("="*50)
    print(f" 涵盖报道 (Total Documents): {total_documents:15,d} 篇")
    print(f" 总计字数 (Total Characters): {chars_in_hundred_millions:16.2f} 亿字")
    print("="*50)

if __name__ == "__main__":
    main()
