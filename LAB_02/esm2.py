"""
Task 2 - File 1: ESM-2 Model and Sequence Embedder
====================================================
This file does ONE thing only:
  Load ESM-2 650M from HuggingFace and embed protein sequences.

It outputs raw per-residue embeddings (layer 33) for every sequence
in each curated dataset. These raw embeddings are saved to disk.
File 2 (feature_extraction.py) then reads these and computes features.

Why separate?
  ESM-2 650M is a 1.3GB model. Loading and running it is the slow,
  heavy part. Once you have the raw embeddings saved, you can run
  feature_extraction.py as many times as you want instantly.

Install:
  pip install torch transformers tqdm pandas numpy

Model used:
  facebook/esm2_t33_650M_UR50D
    - 650 million parameters
    - 33 transformer layers
    - 1,280-dimensional hidden states per residue per layer
    - Trained on UniRef50 with masked language modelling
    - Standard choice for mutation effect and protein classification tasks
    - Reference: Lin Z et al. (2023) Science 379(6637):1123-1130

Input:
  ./curated/PI_curated.csv
  ./curated/NRTI_curated.csv
  ./curated/NNRTI_curated.csv
  ./curated/INI_curated.csv
  (Output of Task 1 - must have a 'sequence' column)

Output:
  ./embeddings/PI_embeddings_raw.npy      shape: (n_seqs, seq_len, 1280)
  ./embeddings/NRTI_embeddings_raw.npy
  ./embeddings/NNRTI_embeddings_raw.npy
  ./embeddings/INI_embeddings_raw.npy
  ./embeddings/{CLASS}_seqids.npy         sequence IDs in same order
  ./embeddings/{CLASS}_sequences.npy      sequences in same order
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
CURATED_DIR   = './curated'
EMBEDDINGS_DIR = './embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
# esm2_t33_650M_UR50D means:
#   t33  = 33 transformer layers
#   650M = 650 million parameters
#   UR50D = trained on UniRef50 database
MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'

# ── Settings ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 8     # reduce to 4 if you get out-of-memory errors on GPU
                   # increase to 16 if you have a large GPU (24GB+)
                   # on CPU, batch size does not matter much for memory

# Which layer to extract from.
# 33 = last layer = richest contextual representation
# This is the standard for downstream classification tasks
REPR_LAYER = 33

# Maximum sequence length ESM-2 can handle
# ESM-2 supports up to 1022 amino acids
# Your longest sequences are RT at 560 aa - well within limit
MAX_SEQ_LEN = 1022

DATASETS = {
    'PI':    'PI_curated.csv',
    'NRTI':  'NRTI_curated.csv',
    'NNRTI': 'NNRTI_curated.csv',
    'INI':   'INI_curated.csv',
}

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')


def clean_sequence(seq):
    """
    Ensure sequence contains only standard amino acids.

    ESM-2 vocabulary: 20 standard amino acids + special tokens.
    X (unknown) is in the vocabulary but contributes low-quality signal.
    Our Task 1 sequences should already be clean - this is a safety check.

    Any non-standard character is replaced with the most common amino acid
    at that position class (we use 'G' glycine as a neutral placeholder).
    In practice your sequences should not need this after Task 1 curation.
    """
    cleaned = []
    for aa in seq.upper():
        if aa in VALID_AAS:
            cleaned.append(aa)
        elif aa == 'X':
            cleaned.append('G')   # ESM-2 handles X but G is cleaner
        else:
            cleaned.append('G')   # fallback for any other character
    return ''.join(cleaned)


def load_model_and_tokenizer():
    """
    Load ESM-2 650M from HuggingFace.

    First run: downloads ~1.3GB to HuggingFace cache (~/.cache/huggingface/).
    Subsequent runs: loads instantly from cache.

    Model is loaded in eval() mode with no_grad() during inference.
    No fine-tuning. Weights are frozen. We only use ESM-2 as a feature extractor.
    """
    print(f"Loading ESM-2 model: {MODEL_NAME}")
    print("First run will download ~1.3GB. This is cached after the first time.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon
        print("Apple Silicon GPU (MPS) detected")
    else:
        device = torch.device('cpu')
        print("No GPU found. Running on CPU. This will be slow (~2-5 min per dataset).")
        print("Tip: reduce BATCH_SIZE to 4 if you see memory issues on CPU.")

    model = model.to(device)
    model.eval()   # inference mode: no dropout, no gradient tracking

    print(f"Model loaded on: {device}")
    print(f"Model parameters: 650M | Hidden size: 1,280 | Layers: 33")
    return tokenizer, model, device


def embed_sequences(sequences, seq_ids, tokenizer, model, device, dataset_label):
    """
    Run ESM-2 on all sequences and extract per-residue embeddings from layer 33.

    Process:
      1. Batch sequences together
      2. Tokenise each batch (adds <cls> and <eos> special tokens automatically)
      3. Run through ESM-2 forward pass
      4. Extract hidden states from layer 33
      5. Remove special token positions (<cls> at index 0, <eos> at last index)
      6. Store as numpy array

    Returns:
      embeddings_list: list of numpy arrays, one per sequence
                       each array has shape (seq_len, 1280)
                       seq_len varies per sequence (no padding in stored output)
    """
    all_embeddings = []
    n = len(sequences)

    print(f"\n  Embedding {n} sequences for {dataset_label}...")
    print(f"  Batch size: {BATCH_SIZE} | Layer: {REPR_LAYER} | Dim: 1,280")

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc=f"  {dataset_label}"):
        batch_seqs = sequences[batch_start : batch_start + BATCH_SIZE]

        # Tokenise batch
        # padding=True pads shorter sequences in the batch to the same length
        # truncation=True truncates sequences longer than MAX_SEQ_LEN
        # return_tensors='pt' returns PyTorch tensors
        inputs = tokenizer(
            batch_seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN + 2,  # +2 for <cls> and <eos> special tokens
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # output_hidden_states=True returns embeddings from ALL 33 layers
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 = embedding layer (before transformer layers)
        # Index 1 = after layer 1
        # Index 33 = after layer 33 (what we want)
        # Shape of each: (batch_size, padded_seq_len, 1280)
        hidden_states = outputs.hidden_states[REPR_LAYER]

        # Move back to CPU for numpy conversion
        hidden_states = hidden_states.cpu().numpy()
        attention_mask = inputs['attention_mask'].cpu().numpy()

        # Process each sequence in the batch
        for i, seq in enumerate(batch_seqs):
            # attention_mask tells us which positions are real vs padding
            # mask = 1 for real tokens (including <cls> and <eos>)
            # mask = 0 for padding tokens

            mask = attention_mask[i]  # shape: (padded_seq_len,)
            seq_len_with_special = mask.sum()  # total real tokens including <cls>/<eos>

            # Extract only real token positions
            emb = hidden_states[i, :seq_len_with_special, :]  # (seq_len+2, 1280)

            # Remove <cls> token (position 0) and <eos> token (last position)
            # These are special tokens added by the tokeniser
            # After removal: shape is (actual_seq_len, 1280)
            emb = emb[1:-1, :]  # remove first and last

            # Verify length matches original sequence
            actual_seq_len = len(seq)
            if emb.shape[0] != actual_seq_len:
                # Handle edge case: truncation may have shortened the sequence
                # Trim embedding to match sequence length
                emb = emb[:actual_seq_len, :]

            all_embeddings.append(emb)

    print(f"  Done. Embedded {len(all_embeddings)} sequences.")
    return all_embeddings


def save_embeddings(embeddings_list, seq_ids, sequences, dataset_label):
    """
    Save embeddings to disk.

    Saves:
      {label}_embeddings_raw.npy  : numpy object array of (n_seqs,) where each
                                    element is a (seq_len, 1280) float32 array
                                    Stored as object array because sequences
                                    have different lengths
      {label}_seqids.npy          : sequence IDs in same order
      {label}_sequences.npy       : cleaned sequences in same order

    In feature_extraction.py (File 2), these are loaded and processed
    into fixed-dimension feature vectors.
    """
    # Save as object array to handle variable sequence lengths
    emb_array = np.empty(len(embeddings_list), dtype=object)
    for i, emb in enumerate(embeddings_list):
        emb_array[i] = emb.astype(np.float32)

    emb_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_label}_embeddings_raw.npy')
    ids_path  = os.path.join(EMBEDDINGS_DIR, f'{dataset_label}_seqids.npy')
    seq_path  = os.path.join(EMBEDDINGS_DIR, f'{dataset_label}_sequences.npy')

    np.save(emb_path, emb_array, allow_pickle=True)
    np.save(ids_path,  np.array(seq_ids),   allow_pickle=False)
    np.save(seq_path,  np.array(sequences),  allow_pickle=True)

    print(f"  Saved: {emb_path}")
    print(f"  Shape check: {len(embeddings_list)} sequences, "
          f"first has shape {embeddings_list[0].shape}")


def process_dataset(label, filename, tokenizer, model, device):
    """Process one drug class dataset end to end."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {label}")
    print(f"{'='*60}")

    path = os.path.join(CURATED_DIR, filename)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run Task 1 first.")
        return

    df = pd.read_csv(path, low_memory=False)

    if 'sequence' not in df.columns:
        print(f"  ERROR: 'sequence' column not found in {filename}")
        return

    # Drop rows where sequence is missing
    df = df.dropna(subset=['sequence']).reset_index(drop=True)
    print(f"  Loaded {len(df)} sequences")

    # Extract and clean sequences
    sequences = [clean_sequence(str(s)) for s in df['sequence'].tolist()]
    seq_ids   = df['SeqID'].tolist() if 'SeqID' in df.columns \
                else list(range(len(df)))

    # Filter out any sequences that ended up empty after cleaning
    valid = [(s, sid) for s, sid in zip(sequences, seq_ids) if len(s) > 0]
    sequences = [v[0] for v in valid]
    seq_ids   = [v[1] for v in valid]

    print(f"  Sequence lengths: min={min(len(s) for s in sequences)}, "
          f"max={max(len(s) for s in sequences)}, "
          f"mean={sum(len(s) for s in sequences)/len(sequences):.0f}")

    # Run ESM-2
    embeddings = embed_sequences(
        sequences, seq_ids, tokenizer, model, device, label
    )

    # Save raw embeddings
    save_embeddings(embeddings, seq_ids, sequences, label)


def main():
    print("\nTask 2 - File 1: ESM-2 Sequence Embedding")
    print("=" * 60)
    print("Model: facebook/esm2_t33_650M_UR50D")
    print("Layer: 33 (last layer, richest contextual representation)")
    print("Output: per-residue embeddings (seq_len x 1280) per sequence")
    print("=" * 60)

    # Load model once — reuse for all four datasets
    tokenizer, model, device = load_model_and_tokenizer()

    for label, filename in DATASETS.items():
        process_dataset(label, filename, tokenizer, model, device)

    print(f"\n{'='*60}")
    print("All datasets embedded.")
    print(f"Raw embeddings saved to: {os.path.abspath(EMBEDDINGS_DIR)}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
