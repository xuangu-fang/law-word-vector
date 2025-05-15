#!/usr/bin/env python
# coding: utf-8

"""
Word vector fine-tuning script for the People's Daily corpus (1949-1979).

This script performs the following steps:
1. Loads a pre-trained Chinese word vector model.
2. Defines time periods for 1949-1979 (consistent with preprocessing).
3. For each period:
    a. Loads the preprocessed corpus for that period.
    b. Creates a new Word2Vec model, initializing its vocabulary with the period's corpus 
       and intersecting with the pre-trained model's vectors.
    c. Fine-tunes the model on the period's corpus.
    d. Saves the fine-tuned model.
4. Includes a basic test function to check word similarities in the fine-tuned models.
"""

import logging
from pathlib import Path
import os

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# --- Configuration ---
# Path to the pre-trained word vectors
PRETRAINED_VECTORS_PATH = Path.home() / "gensim-data" / "vectors" / "chinese_vectors.kv"

# Directory containing the processed period corpora (from preprocess_corpus_1949_1979.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_CORPUS_DIR = PROJECT_ROOT / "processed_data" / "corpus_1949-1979"

# Output directory for fine-tuned models
FINETUNED_MODELS_DIR = PROJECT_ROOT / "models" / "fine_tuned_vectors_1949-1979"

# Time periods (must match those in the preprocessing script)
TIME_PERIODS = {
    "1949-1956": (1949, 1956),
    "1957-1965": (1957, 1965),
    "1966-1976": (1966, 1976),
    "1977-1979": (1977, 1979)
}

# Word2Vec Fine-tuning Parameters
VECTOR_SIZE = 300 # Should match the dimensionality of the pre-trained model if possible
WINDOW_SIZE = 5
MIN_COUNT = 5     # Minimum frequency for a word to be included in the vocabulary
WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
EPOCHS = 10        # Number of epochs for fine-tuning on each period's corpus
SG = 1             # 0 for CBOW, 1 for Skip-gram (Skip-gram often better for fine-tuning)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Classes and Functions ---

class EpochLogger(CallbackAny2Vec):
    """Callback to log information after each epoch."""
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.info(f"Epoch {self.epoch+1} finished. Loss: {loss}")
        self.epoch += 1

class CorpusSentenceIterator:
    """An iterator that reads sentences from a preprocessed corpus file."""
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()

def check_vector_dimensions(keyed_vectors):
    """Checks and logs the dimensionality of the loaded KeyedVectors."""
    if keyed_vectors:
        try:
            # Access a vector to determine its size
            sample_word = keyed_vectors.index_to_key[0]
            dim = keyed_vectors[sample_word].shape[0]
            logging.info(f"Loaded pre-trained vectors with dimensionality: {dim}")
            return dim
        except Exception as e:
            logging.error(f"Could not determine dimensionality of pre-trained vectors: {e}")
            return None
    return None

# --- Main Fine-tuning Logic ---
def main():
    FINETUNED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Fine-tuned models will be saved to: {FINETUNED_MODELS_DIR}")

    # 1. Load Pre-trained Vectors
    if not PRETRAINED_VECTORS_PATH.exists():
        logging.error(f"Pre-trained vectors not found at: {PRETRAINED_VECTORS_PATH}")
        logging.error("Please ensure you have downloaded the 'chinese_vectors.kv' model or adjust PRETRAINED_VECTORS_PATH.")
        return
    
    logging.info(f"Loading pre-trained vectors from: {PRETRAINED_VECTORS_PATH}")
    try:
        pretrained_kv = KeyedVectors.load(str(PRETRAINED_VECTORS_PATH), mmap='r')
        logging.info(f"Successfully loaded pre-trained vectors. Vocabulary size: {len(pretrained_kv.index_to_key)}")
        pretrained_vector_size = check_vector_dimensions(pretrained_kv)
        if pretrained_vector_size is None:
            logging.error("Cannot proceed without knowing pre-trained vector dimensionality.")
            return
        # Adjust VECTOR_SIZE to match pre-trained if not already set, or warn if different
        global VECTOR_SIZE
        if VECTOR_SIZE != pretrained_vector_size:
            logging.warning(f"Global VECTOR_SIZE ({VECTOR_SIZE}) differs from pre-trained model ({pretrained_vector_size}). Using pre-trained model's dimensionality.")
            VECTOR_SIZE = pretrained_vector_size

    except Exception as e:
        logging.error(f"Error loading pre-trained vectors: {e}")
        return

    # 2. Fine-tune for each period
    # We will use an incremental fine-tuning approach: 
    # The model from period N is used as the starting point for period N+1.
    current_model_kv = pretrained_kv # Start with the general pre-trained model
    
    sorted_periods = sorted(TIME_PERIODS.keys())

    for period_name in sorted_periods:
        logging.info(f"\n--- Fine-tuning for period: {period_name} ---")
        corpus_file = PROCESSED_CORPUS_DIR / f"{period_name}.txt"

        if not corpus_file.exists():
            logging.warning(f"Corpus file not found for period {period_name}: {corpus_file}. Skipping.")
            continue

        logging.info(f"Loading corpus sentences from: {corpus_file}")
        sentences = CorpusSentenceIterator(corpus_file)

        # Create a new Word2Vec model for fine-tuning
        # Initialize with the vocabulary of the current period's corpus
        model = Word2Vec(
            vector_size=VECTOR_SIZE,
            window=WINDOW_SIZE,
            min_count=MIN_COUNT,
            workers=WORKERS,
            sg=SG,
            epochs=EPOCHS, # This will be used in model.train()
            callbacks=[EpochLogger()]
        )

        logging.info("Building vocabulary for the current period...")
        model.build_vocab(sentences)
        logging.info(f"Vocabulary built for {period_name}. New words: {len(model.wv.index_to_key)}")

        # Intersect vocabulary with the current_model_kv (either pre-trained or from previous period)
        # This transfers the learned vectors for words present in both vocabularies.
        # Gensim's `model.wv.vectors_lockf` can be used for more advanced locking strategies,
        # but a simple intersection and further training is a common fine-tuning approach.
        logging.info("Intersecting vocabulary with current model's vectors...")
        
        # Gensim 4.0.0+ way: model.wv.vectors_lockf & model.wv.intersect_word2vec_format
        # For a simpler approach: manually copy vectors for intersection, or use `Word2Vec.load_word2vec_format` with `lockf`
        # Here, we build vocab from new corpus, then update with pre-trained weights.
        
        # Get the intersection of words
        intersecting_words_count = 0
        for word in model.wv.index_to_key:
            if word in current_model_kv:
                model.wv[word] = current_model_kv[word]
                intersecting_words_count += 1
        logging.info(f"Copied {intersecting_words_count} vectors from the current model to the new model for {period_name}.")

        # Fine-tune the model on the current period's corpus
        logging.info(f"Starting fine-tuning for {period_name} with {model.epochs} epochs...")
        model.train(
            sentences, 
            total_examples=model.corpus_count, 
            epochs=model.epochs, # Use the epochs defined in model init
            start_alpha=model.alpha, # learning rate
            end_alpha=model.min_alpha # final learning rate
        )
        logging.info(f"Fine-tuning finished for {period_name}.")

        # Save the fine-tuned KeyedVectors (the embedding layer)
        fine_tuned_kv_path = FINETUNED_MODELS_DIR / f"{period_name}_wordvectors.kv"
        model.wv.save(str(fine_tuned_kv_path))
        logging.info(f"Fine-tuned KeyedVectors for {period_name} saved to: {fine_tuned_kv_path}")

        # Update current_model_kv for the next iteration, so fine-tuning is incremental
        current_model_kv = model.wv 
        logging.info(f"Model for {period_name} will be the base for the next period.")

    logging.info("\nAll periods fine-tuned.")

    # 3. Basic Test of Fine-tuned Models
    test_keywords = ["法治", "法制", "宪法", "改革", "开放", "民主"]
    logging.info(f"\n--- Testing fine-tuned models with keywords: {test_keywords} ---")
    for period_name in sorted_periods:
        model_path = FINETUNED_MODELS_DIR / f"{period_name}_wordvectors.kv"
        if model_path.exists():
            logging.info(f"\nTesting model for period: {period_name}")
            try:
                period_model_kv = KeyedVectors.load(str(model_path))
                for word in test_keywords:
                    if word in period_model_kv:
                        similar_words = period_model_kv.most_similar(word, topn=5)
                        logging.info(f"  Most similar to '{word}' in {period_name}: {similar_words}")
                    else:
                        logging.info(f"  Word '{word}' not in vocabulary for {period_name}.")
            except Exception as e:
                logging.error(f"Error testing model for {period_name}: {e}")
        else:
            logging.warning(f"Fine-tuned model not found for {period_name} at {model_path}. Cannot test.")

if __name__ == "__main__":
    main() 