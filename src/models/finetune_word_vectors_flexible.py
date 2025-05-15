#!/usr/bin/env python
# coding: utf-8

"""
Flexible Word Vector Fine-tuning Script for People's Daily Corpus.

This script:
1. Loads a pre-trained Chinese word vector model.
2. Takes a list of period definitions (name, start_year, end_year).
3. Uses CorpusManager to get or create period-specific corpora.
4. Fine-tunes Word2Vec models for each defined period.
5. Supports both incremental fine-tuning (default) and independent fine-tuning from the base pre-trained model.
6. Saves the fine-tuned models and performs basic similarity tests.
"""

import logging
from pathlib import Path
import os
import sys

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# --- Adjust sys.path to import CorpusManager from sibling directory --- 
# This assumes finetune_word_vectors_flexible.py is in src/models/
# and corpus_manager.py is in src/data/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SRC_DIR = SCRIPT_DIR.parent
DATA_MODULE_DIR = PROJECT_SRC_DIR / "data"
sys.path.append(str(PROJECT_SRC_DIR))

from data.corpus_manager import CorpusManager # type: ignore

# --- Configuration ---
PRETRAINED_VECTORS_PATH = Path.home() / "gensim-data" / "vectors" / "chinese_vectors.kv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINETUNED_MODELS_OUTPUT_DIR = PROJECT_ROOT / "models" / "fine_tuned_vectors_flexible"

# --- Periods Definition (Customize this list) ---
# Each dictionary should have "name", "start_year", and "end_year"
PERIODS_TO_FINETUNE = [
    {"name": "Era1_1949-1956", "start_year": 1949, "end_year": 1956},
    {"name": "Era2_1957-1965", "start_year": 1957, "end_year": 1965},
    {"name": "Era3_1966-1976", "start_year": 1966, "end_year": 1976},
    {"name": "Era4_1977-1979", "start_year": 1977, "end_year": 1979},
    # Example for later periods if data is available:
    # {"name": "Era5_1980-1989", "start_year": 1980, "end_year": 1989},
    # {"name": "FullRange_1949-2023", "start_year": 1949, "end_year": 2023} 
]

# --- Fine-tuning Strategy ---
# If True, model from period N is used as base for period N+1 (for first period, uses PRETRAINED_VECTORS_PATH).
# If False, each period fine-tunes independently from PRETRAINED_VECTORS_PATH.
INCREMENTAL_FINETUNING = True

# If True, period corpora will be recreated even if they already exist.
FORCE_CREATE_PERIOD_CORPORA = False 

# --- Word2Vec Fine-tuning Parameters ---
# VECTOR_SIZE will be dynamically set from the pre-trained model
WINDOW_SIZE = 5
MIN_COUNT = 5     
WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
EPOCHS = 10        
SG = 1 # 0 for CBOW, 1 for Skip-gram

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Classes and Functions ---

class EpochLogger(CallbackAny2Vec):
    def __init__(self, period_name):
        self.epoch = 0
        self.period_name = period_name
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.info(f"Period '{self.period_name}' - Epoch {self.epoch+1}/{model.epochs} finished. Loss: {loss}")
        self.epoch += 1

class PeriodCorpusSentenceIterator:
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()

def get_pretrained_vectors_info(kv_path: Path):
    if not kv_path.exists():
        logging.error(f"Pre-trained vectors not found: {kv_path}")
        return None, 0
    try:
        kv = KeyedVectors.load(str(kv_path), mmap='r')
        vocab_size = len(kv.index_to_key)
        vector_dim = kv.vector_size
        logging.info(f"Loaded pre-trained vectors from {kv_path}. Vocab: {vocab_size}, Dim: {vector_dim}")
        return kv, vector_dim
    except Exception as e:
        logging.error(f"Error loading pre-trained vectors from {kv_path}: {e}")
        return None, 0

# --- Main Fine-tuning Logic ---
def main():
    FINETUNED_MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Fine-tuned models will be saved to: {FINETUNED_MODELS_OUTPUT_DIR}")

    corpus_manager = CorpusManager() # Uses default paths from corpus_manager.py

    # 1. Load Base Pre-trained Vectors
    base_pretrained_kv, initial_vector_size = get_pretrained_vectors_info(PRETRAINED_VECTORS_PATH)
    if not base_pretrained_kv or initial_vector_size == 0:
        logging.error("Could not load base pre-trained vectors. Exiting.")
        return
    
    current_base_kv_for_tuning = base_pretrained_kv

    # Ensure periods are sorted if incremental, though explicit definition order is usually fine
    # sorted_periods_config = sorted(PERIODS_TO_FINETUNE, key=lambda p: p['start_year'])
    # Using the defined order from PERIODS_TO_FINETUNE directly
    
    for period_config in PERIODS_TO_FINETUNE:
        period_name = period_config["name"]
        start_year = period_config["start_year"]
        end_year = period_config["end_year"]
        
        logging.info(f"\n========== Processing Period: {period_name} ({start_year}-{end_year}) ==========")

        # 2. Get or Create Period-Specific Corpus
        try:
            period_corpus_file = corpus_manager.get_or_create_period_corpus(
                period_name=period_name, 
                start_year=start_year, 
                end_year=end_year,
                force_create=FORCE_CREATE_PERIOD_CORPORA
            )
            logging.info(f"Using corpus file for {period_name}: {period_corpus_file}")
        except FileNotFoundError as e:
            logging.error(f"Cannot proceed with period {period_name}: {e}. Ensure yearly data exists and preprocess_yearly_data.py has run.")
            continue # Skip to next period
        except Exception as e:
            logging.error(f"Unexpected error getting corpus for {period_name}: {e}. Skipping period.")
            continue
            
        sentences = PeriodCorpusSentenceIterator(period_corpus_file)

        # 3. Initialize and Fine-tune Word2Vec Model
        model = Word2Vec(
            vector_size=initial_vector_size, # Use dim from loaded pre-trained model
            window=WINDOW_SIZE,
            min_count=MIN_COUNT,
            workers=WORKERS,
            sg=SG,
            epochs=EPOCHS, 
            callbacks=[EpochLogger(period_name)]
        )

        logging.info(f"Building vocabulary for '{period_name}' from {period_corpus_file}...")
        model.build_vocab(sentences) # Build vocab from the current period's specific corpus
        logging.info(f"Vocabulary built for '{period_name}'. New words in corpus: {len(model.wv.index_to_key)}")

        # Determine which KeyedVectors to use for weight initialization
        source_kv_for_init = base_pretrained_kv if not INCREMENTAL_FINETUNING else current_base_kv_for_tuning
        source_kv_name = "original pre-trained" if not INCREMENTAL_FINETUNING or current_base_kv_for_tuning == base_pretrained_kv else "previous period's model"
        
        logging.info(f"Initializing model weights for '{period_name}' using vectors from {source_kv_name} model.")
        intersecting_words_count = 0
        for word in model.wv.index_to_key:
            if word in source_kv_for_init:
                model.wv[word] = source_kv_for_init[word]
                intersecting_words_count += 1
        logging.info(f"Copied {intersecting_words_count} vectors from {source_kv_name} model to the new model for '{period_name}'.")
        
        # Update vocabulary with vectors from the source_kv_for_init, this handles intersection.
        # model.wv.intersect_word2vec_format(source_kv_for_init_path, lockf=0.0) # Alternative if source_kv is a file
        # The manual loop above is more explicit for KeyedVector objects.

        logging.info(f"Starting fine-tuning for '{period_name}' with {model.epochs} epochs...")
        model.train(
            sentences, 
            total_examples=model.corpus_count, 
            epochs=model.epochs,
            start_alpha=model.alpha, 
            end_alpha=model.min_alpha 
        )
        logging.info(f"Fine-tuning finished for '{period_name}'.")

        # Save the fine-tuned KeyedVectors
        fine_tuned_kv_path = FINETUNED_MODELS_OUTPUT_DIR / f"{period_name}_wordvectors.kv"
        model.wv.save(str(fine_tuned_kv_path))
        logging.info(f"Fine-tuned KeyedVectors for '{period_name}' saved to: {fine_tuned_kv_path}")

        if INCREMENTAL_FINETUNING:
            current_base_kv_for_tuning = model.wv 
            logging.info(f"Model for '{period_name}' will be the base for the next period (if any). VSize: {len(current_base_kv_for_tuning.index_to_key)}")

    logging.info("\n========== All specified periods fine-tuned. ==========")

    # 4. Basic Test of All Fine-tuned Models Generated in this Run
    test_keywords = ["法治", "法制", "宪法", "改革", "开放", "民主", "社会主义", "市场经济"]
    logging.info(f"\n--- Testing fine-tuned models with keywords: {test_keywords} ---")
    for period_config in PERIODS_TO_FINETUNE:
        period_name = period_config["name"]
        model_path = FINETUNED_MODELS_OUTPUT_DIR / f"{period_name}_wordvectors.kv"
        if model_path.exists():
            logging.info(f"\n--- Results for Period: {period_name} ---")
            try:
                period_model_kv = KeyedVectors.load(str(model_path))
                for word in test_keywords:
                    if word in period_model_kv:
                        similar_words = period_model_kv.most_similar(word, topn=5)
                        logging.info(f"  Most similar to '{word}': {similar_words}")
                    else:
                        logging.info(f"  Word '{word}' not in vocabulary for {period_name}.")
            except Exception as e:
                logging.error(f"Error testing model for {period_name}: {e}")
        else:
            # This might happen if corpus creation failed for a period earlier.
            logging.warning(f"Fine-tuned model not found for {period_name} at {model_path} for testing. It might have been skipped.")

if __name__ == "__main__":
    # Before running, ensure:
    # 1. `src/data/by_year_preprocess.py` has been run to generate yearly corpus files.
    # 2. `PRETRAINED_VECTORS_PATH` points to your base `chinese_vectors.kv`.
    # 3. `PERIODS_TO_FINETUNE` list at the top of this script is configured as desired.
    main() 