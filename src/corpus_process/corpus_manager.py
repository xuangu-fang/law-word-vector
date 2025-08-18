#!/usr/bin/env python
# coding: utf-8

"""
Corpus Manager for combining yearly preprocessed data into period-specific corpora.
"""

import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# --- Configuration (Directories should align with other scripts) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YEARLY_CORPUS_DIR = PROJECT_ROOT / "processed_data" / "yearly_corpus" # Input: Where year_*.txt files are
PERIOD_CORPUS_DIR = PROJECT_ROOT / "processed_data" / "period_corpus" # Output: Where period_*.txt files will be saved
STATS_DIR = PROJECT_ROOT / "processed_data" / "stats"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CorpusManager:
    def __init__(self, yearly_corpus_dir=YEARLY_CORPUS_DIR, period_corpus_dir=PERIOD_CORPUS_DIR, stats_dir=STATS_DIR):
        self.yearly_corpus_dir = Path(yearly_corpus_dir)
        self.period_corpus_dir = Path(period_corpus_dir)
        self.stats_dir = Path(stats_dir)
        
        self.period_corpus_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"CorpusManager initialized.")
        logging.info(f"  Yearly corpus input directory: {self.yearly_corpus_dir}")
        logging.info(f"  Period corpus output directory: {self.period_corpus_dir}")

    def create_period_corpus(self, period_name: str, start_year: int, end_year: int) -> Path:
        """
        Creates a single corpus file for a defined period by concatenating yearly files.

        Args:
            period_name (str): A descriptive name for the period (e.g., "1949-1978").
                               This will be used as the filename (period_name.txt).
            start_year (int): The starting year of the period (inclusive).
            end_year (int): The ending year of the period (inclusive).

        Returns:
            Path: The path to the generated period corpus file.
        
        Raises:
            FileNotFoundError: If any of the required yearly corpus files are missing.
        """
        period_corpus_file_path = self.period_corpus_dir / f"{period_name}.txt"
        logging.info(f"Attempting to create corpus for period '{period_name}' ({start_year}-{end_year}) at {period_corpus_file_path}")

        yearly_files_to_combine = []
        for year in range(start_year, end_year + 1):
            year_file = self.yearly_corpus_dir / f"{year}.txt"
            if not year_file.exists():
                logging.error(f"Missing yearly corpus file for {year}: {year_file}")
                raise FileNotFoundError(f"Missing yearly corpus file for {year}: {year_file}")
            yearly_files_to_combine.append(year_file)
        
        logging.info(f"Found {len(yearly_files_to_combine)} yearly files to combine for period '{period_name}'.")

        with open(period_corpus_file_path, 'w', encoding='utf-8') as outfile:
            for year_file_path in tqdm(yearly_files_to_combine, desc=f"Combining for {period_name}"):
                with open(year_file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
        
        logging.info(f"Successfully created period corpus: {period_corpus_file_path}")
        return period_corpus_file_path

    def get_or_create_period_corpus(self, period_name: str, start_year: int, end_year: int, force_create: bool = False) -> Path:
        """
        Retrieves the path to a period-specific corpus file. If it doesn't exist or force_create is True,
        it creates the corpus by combining preprocessed yearly files.

        Args:
            period_name (str): A descriptive name for the period (e.g., "1949-1978").
            start_year (int): The starting year of the period (inclusive).
            end_year (int): The ending year of the period (inclusive).
            force_create (bool): If True, the period corpus will be recreated even if it already exists.
                                 Defaults to False.

        Returns:
            Path: The path to the period corpus file.
        """
        period_corpus_file_path = self.period_corpus_dir / f"{period_name}.txt"

        if period_corpus_file_path.exists() and not force_create:
            logging.info(f"Period corpus for '{period_name}' already exists: {period_corpus_file_path}")
            return period_corpus_file_path
        else:
            if force_create and period_corpus_file_path.exists():
                 logging.info(f"force_create is True. Recreating period corpus for '{period_name}'.")
            return self.create_period_corpus(period_name, start_year, end_year)

    def calculate_and_save_period_stats(self, periods_config: list):
        """
        Calculates and saves statistics for a list of defined periods.
        Assumes period corpus files have been created by get_or_create_period_corpus.

        Args:
            periods_config (list): A list of dictionaries, where each dictionary defines a period.
                                   Example: [{"name": "1949-1978", "start": 1949, "end": 1978}, ...]
        """
        period_stats_summary = []
        for period_info in periods_config:
            period_name = period_info["name"]
            period_file_path = self.period_corpus_dir / f"{period_name}.txt"
            
            logging.info(f"Calculating stats for period: {period_name}")
            if not period_file_path.exists():
                logging.warning(f"Corpus file for period '{period_name}' not found at {period_file_path}. Cannot calculate stats.")
                period_stats_summary.append({
                    "Period Name": period_name,
                    "Start Year": period_info.get("start", "N/A"),
                    "End Year": period_info.get("end", "N/A"),
                    "Processed Documents": 0,
                    "Total Tokens": 0,
                    "Avg Tokens/Doc": 0,
                    "Vocabulary Size": 0,
                    "Error": "Corpus file not found"
                })
                continue

            processed_docs = 0
            total_tokens = 0
            all_tokens_in_period = []
            
            with open(period_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    if tokens: # count non-empty lines as documents
                        processed_docs += 1
                        total_tokens += len(tokens)
                        all_tokens_in_period.extend(tokens)
            
            avg_tokens_per_doc = total_tokens / processed_docs if processed_docs > 0 else 0
            vocabulary_size = len(set(all_tokens_in_period)) if all_tokens_in_period else 0
            
            period_stats_summary.append({
                "Period Name": period_name,
                "Start Year": period_info.get("start", "N/A"),
                "End Year": period_info.get("end", "N/A"),
                "Processed Documents": processed_docs,
                "Total Tokens": total_tokens,
                "Avg Tokens/Doc": round(avg_tokens_per_doc, 2),
                "Vocabulary Size": vocabulary_size,
                "Error": ""
            })
            logging.info(f"Stats for {period_name}: Docs={processed_docs}, Tokens={total_tokens}, Vocab={vocabulary_size}")
        
        if not period_stats_summary:
            logging.info("No period statistics to save.")
            return

        stats_df = pd.DataFrame(period_stats_summary)
        stats_df.set_index("Period Name", inplace=True)
        
        # Generate a filename based on the number of periods or a hash of period names
        num_periods = len(periods_config)
        example_period_name = periods_config[0]["name"].replace("-","to") if periods_config else "custom"
        stats_filename = f"period_corpus_stats_{num_periods}periods_starting_{example_period_name}.csv"
        stats_file_path = self.stats_dir / stats_filename
        
        stats_df.to_csv(stats_file_path)
        logging.info(f"Period Corpus Statistics saved to {stats_file_path}")
        print("\nPeriod Corpus Statistics:")
        print(stats_df)

# --- Example Usage (can be run as a script for testing) ---
def main():
    logging.info("Running CorpusManager example...")
    manager = CorpusManager()

    # Define example periods based on your request
    example_periods = [
        {"name": "1949-1978", "start": 1949, "end": 1978},
        {"name": "1979-1992", "start": 1979, "end": 1992}, # Example: Reform and Opening Up
        {"name": "1993-2002", "start": 1993, "end": 2002}, # Example: Socialist Market Economy
        {"name": "2003-2012", "start": 2003, "end": 2012}, # Example: Scientific Development
        {"name": "2013-2023", "start": 2013, "end": 2023}  # Example: New Era
    ]

    # Ensure yearly data exists for these ranges by running preprocess_yearly_data.py first for 1949-2023.
    # For this example, we assume it has run.
    
    for period_info in example_periods:
        try:
            corpus_path = manager.get_or_create_period_corpus(
                period_name=period_info["name"],
                start_year=period_info["start"],
                end_year=period_info["end"],
                force_create=False # Set to True to always recreate
            )
            logging.info(f"Corpus for period {period_info['name']} is ready at: {corpus_path}")
        except FileNotFoundError as e:
            logging.error(f"Could not create/get corpus for {period_info['name']}: {e}")
            logging.error("Please ensure preprocess_yearly_data.py has been run for the required years.")
        except Exception as e:
            logging.error(f"An unexpected error occurred for period {period_info['name']}: {e}")

    # Calculate and save stats for these periods
    try:
        manager.calculate_and_save_period_stats(example_periods)
    except Exception as e:
        logging.error(f"Error calculating period stats: {e}")

if __name__ == "__main__":
    # This main function is for demonstration and testing of CorpusManager.
    # In a real pipeline, you would import CorpusManager and use its methods
    # in your fine-tuning script or a master orchestration script.
    
    # Before running this example, ensure you have run preprocess_yearly_data.py
    # for the years covered in `example_periods` (e.g., 1949-2023).
    main() 