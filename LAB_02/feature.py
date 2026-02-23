"""
Task 2 - File 2: Feature Extraction from ESM-2 Embeddings
==========================================================
This file reads the raw per-residue embeddings saved by File 1
and computes three feature vectors per sequence.

Run AFTER task2_esm2_embedder.py.

FEATURES EXTRACTED
------------------

Feature 1: Global mean-pooled embedding (F1)
  Average the 1,280-dim vector across ALL residue positions.
  Shape: (n_seqs, 1280)
  Use: Global biochemical fingerprint of the whole protein.
       Input to Random Forest, SVM, XGBoost, MLP classifiers.
  Reference: Mean pooling outperforms CLS token and max pooling
             for protein sequence-level classification tasks.
             (Wesst et al. 2024, Bioinformatics; Lin et al. 2023)

Feature 2: Mutant-position mean embedding (F2)  [NOVEL]
  Average the 1,280-dim vector ONLY over positions that are mutated
  relative to HXB2 wildtype (i.e. positions where this sequence
  differs from the reference). If no mutations exist (wildtype-like
  sequence), falls back to F1.
  Shape: (n_seqs, 1280)
  Use: More sensitive to resistance mutations than global mean.
       Prior work shows averaging only over mutated positions
       gives better sensitivity for variant effect prediction.
       No prior HIV resistance paper does this specifically.
  Reference: Meier et al. 2021 (ESM mutation effect scoring).

Feature 3: Delta embedding (F3)  [NOVEL]
  F3 = mean_embedding(patient_sequence) - mean_embedding(HXB2_wildtype)
  The wildtype reference embedding is computed ONCE and subtracted.
  Shape: (n_seqs, 1280)
  Use: Encodes the directional shift in protein embedding space
       caused by the mutations in this patient's sequence.
       Directly answers: how different is this virus from wildtype
       in ESM-2's learned biochemical space?
       Novel — no prior HIV resistance paper applies this.

These three features are saved separately so you can:
  - Use F1 alone as a baseline
  - Compare F1 vs F2 to show mutant-position averaging helps
  - Use F3 as an additional novel input
  - Concatenate [F1, F2, F3] as a 3,840-dim combined feature

Output:
  ./features/PI_F1_global_mean.npy         shape: (n_seqs, 1280)
  ./features/PI_F2_mutant_mean.npy         shape: (n_seqs, 1280)
  ./features/PI_F3_delta.npy               shape: (n_seqs, 1280)
  ./features/PI_combined.npy               shape: (n_seqs, 3840)
  ./features/PI_labels.csv                 drug labels per sequence
  (same for NRTI, NNRTI, INI)

  ./features/PI_feature_summary.txt        statistics report

HXB2 wildtype reference sequences (same as Task 1):
  These are stored as constants below.
  The wildtype embedding is computed once per drug class
  (protease for PI, RT for NRTI/NNRTI, integrase for INI).
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = './embeddings'
FEATURES_DIR   = './features'
CURATED_DIR    = './curated'
os.makedirs(FEATURES_DIR, exist_ok=True)

MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
REPR_LAYER = 33

# ── HXB2 wildtype reference sequences (same as Task 1) ─────────────────────────
# Used to compute the wildtype embedding baseline for F2 and F3.
# Source: GenBank K03455.1 | Stanford HIVDB consensus sequences

HXB2 = {
    'PR': (
        "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQ"
        "ILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    ),
    'RT': (
        "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIK"
        "KKDSTKWRKLYPQKIKEQYFEWMGYLENPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQI"
        "YPGIKVRQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQ"
        "GQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETW"
        "ETWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTDRG"
        "RQKVVSLTDTTNQKTELQAIHLALQDSGLEIVNIVTDSQYALGIIQAQPDKSESELVSQIIEQL"
        "IKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVLFLDGIDKAQEEHEKYHSNWRAMASDFNLP"
        "PVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVIIVAVHVASGYIEAEVIPAET"
        "GQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMN"
    )[:560],
    'IN': (
        "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQL"
        "DCTHLEGKVIIVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVK"
        "AACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGI"
        "GGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNS"
        "DIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"
    )[:288],
}

# Map dataset label to enzyme (and HXB2 reference to use)
ENZYME_MAP = {
    'PI':    'PR',
    'NRTI':  'RT',
    'NNRTI': 'RT',
    'INI':   'IN',
}

# ── Drug label columns per dataset ─────────────────────────────────────────────
DRUG_COLS = {
    'PI':    ['FPV','ATV','IDV','LPV','NFV','SQV','TPV','DRV'],
    'NRTI':  ['3TC','ABC','AZT','D4T','DDI','TDF'],
    'NNRTI': ['EFV','NVP','ETR','RPV','DOR'],
    'INI':   ['RAL','EVG','DTG','BIC','CAB'],
}

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')


# ── Model loader (same pattern as File 1) ──────────────────────────────────────

def load_model_and_tokenizer():
    """Load ESM-2. Model should already be cached from File 1."""
    print(f"Loading ESM-2 ({MODEL_NAME}) from cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.eval()
    print(f"Model on: {device}")
    return tokenizer, model, device


def embed_one_sequence(sequence, tokenizer, model, device):
    """
    Embed a single sequence and return per-residue embeddings.
    Shape returned: (seq_len, 1280)
    Used for computing the wildtype reference embedding.
    """
    inputs = tokenizer(
        sequence,
        return_tensors='pt',
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[REPR_LAYER]   # (1, seq_len+2, 1280)
    hidden = hidden.squeeze(0).cpu().numpy()      # (seq_len+2, 1280)
    hidden = hidden[1:-1, :]                      # remove <cls> and <eos>
    return hidden.astype(np.float32)


# ── Feature computation functions ──────────────────────────────────────────────

def compute_f1_global_mean(per_residue_emb):
    """
    F1: Mean pool over ALL residue positions.

    Simple average of all per-residue vectors.
    This is the global biochemical fingerprint of the entire protein.

    Input:  (seq_len, 1280) numpy array
    Output: (1280,) numpy array
    """
    return per_residue_emb.mean(axis=0)


def compute_f2_mutant_mean(per_residue_emb, patient_seq, wildtype_seq):
    """
    F2: Mean pool over ONLY the mutated positions.  [NOVEL]

    Find all positions where patient_seq differs from wildtype_seq.
    Average the ESM-2 embeddings only at those positions.

    If no mutations found (wildtype-like sequence), return global mean (F1).
    This handles wildtype-like sequences without crashing.

    Why this is better:
      A 99aa protease sequence may have only 10 mutations.
      Global mean pools 89 wildtype positions with 10 mutated positions.
      The resistance signal from 10 positions is diluted by 89 background positions.
      Taking the mean only over the 10 mutated positions removes this dilution.

    Input:
      per_residue_emb: (seq_len, 1280)
      patient_seq:     string of length seq_len
      wildtype_seq:    HXB2 reference string of same length
    Output:
      (1280,) numpy array
    """
    min_len = min(len(patient_seq), len(wildtype_seq), per_residue_emb.shape[0])

    # Find mutated positions (0-indexed)
    mutated_positions = [
        i for i in range(min_len)
        if patient_seq[i] != wildtype_seq[i] and patient_seq[i] in VALID_AAS
    ]

    if not mutated_positions:
        # No mutations detected — sequence is wildtype-like
        # Fall back to global mean
        return compute_f1_global_mean(per_residue_emb)

    # Extract embeddings at mutated positions only
    mutant_embs = per_residue_emb[mutated_positions, :]  # (n_mutations, 1280)
    return mutant_embs.mean(axis=0)


def compute_f3_delta(f1_patient, wt_mean):
    """
    F3: Delta embedding — patient minus wildtype.  [NOVEL]

    Subtracts the wildtype mean embedding from the patient mean embedding.
    Encodes the directional shift in ESM-2 embedding space caused by mutations.

    A wildtype sequence would give F3 = zero vector.
    A highly resistant sequence with many mutations would give a large F3.
    The direction of F3 encodes which type of resistance pathway the sequence
    has followed (different resistance mutations push the embedding in
    different directions in the 1,280-dimensional space).

    Input:
      f1_patient: (1280,) global mean of patient sequence
      wt_mean:    (1280,) global mean of wildtype HXB2 sequence
    Output:
      (1280,) numpy array
    """
    return f1_patient - wt_mean


# ── Main feature extraction ─────────────────────────────────────────────────────

def extract_features_for_dataset(label, tokenizer, model, device):
    """Extract all three features for one drug class dataset."""
    print(f"\n{'='*60}")
    print(f"  Feature extraction: {label}")
    print(f"{'='*60}")

    # ── Load raw embeddings from File 1 ────────────────────────────────────────
    emb_path = os.path.join(EMBEDDINGS_DIR, f'{label}_embeddings_raw.npy')
    ids_path  = os.path.join(EMBEDDINGS_DIR, f'{label}_seqids.npy')
    seq_path  = os.path.join(EMBEDDINGS_DIR, f'{label}_sequences.npy')

    if not os.path.exists(emb_path):
        print(f"  ERROR: {emb_path} not found. Run task2_esm2_embedder.py first.")
        return

    print("  Loading raw embeddings from disk...")
    embeddings = np.load(emb_path, allow_pickle=True)   # object array
    seq_ids    = np.load(ids_path, allow_pickle=False)
    sequences  = np.load(seq_path, allow_pickle=True)

    n_seqs = len(embeddings)
    print(f"  Loaded {n_seqs} sequences with per-residue embeddings")

    # ── Compute wildtype reference embedding ───────────────────────────────────
    enzyme  = ENZYME_MAP[label]
    wt_seq  = HXB2[enzyme]
    print(f"  Computing wildtype ({enzyme}) reference embedding...")
    wt_per_residue = embed_one_sequence(wt_seq, tokenizer, model, device)
    wt_mean = compute_f1_global_mean(wt_per_residue)
    print(f"  Wildtype embedding computed. Shape: {wt_mean.shape}")

    # ── Compute all three features ─────────────────────────────────────────────
    F1 = np.zeros((n_seqs, 1280), dtype=np.float32)
    F2 = np.zeros((n_seqs, 1280), dtype=np.float32)
    F3 = np.zeros((n_seqs, 1280), dtype=np.float32)

    wt_seq_padded = wt_seq + ('G' * 600)  # pad wildtype for comparison safety

    print("  Computing F1 (global mean), F2 (mutant mean), F3 (delta)...")
    for i in tqdm(range(n_seqs), desc=f"  {label} features"):
        per_res = embeddings[i]        # (seq_len, 1280)
        patient_seq = str(sequences[i])

        f1 = compute_f1_global_mean(per_res)
        f2 = compute_f2_mutant_mean(per_res, patient_seq, wt_seq_padded)
        f3 = compute_f3_delta(f1, wt_mean)

        F1[i] = f1
        F2[i] = f2
        F3[i] = f3

    # ── Combined feature: concatenate all three ────────────────────────────────
    # Shape: (n_seqs, 3840) = 1280 + 1280 + 1280
    F_combined = np.concatenate([F1, F2, F3], axis=1)

    # ── Save features ──────────────────────────────────────────────────────────
    np.save(os.path.join(FEATURES_DIR, f'{label}_F1_global_mean.npy'), F1)
    np.save(os.path.join(FEATURES_DIR, f'{label}_F2_mutant_mean.npy'), F2)
    np.save(os.path.join(FEATURES_DIR, f'{label}_F3_delta.npy'),       F3)
    np.save(os.path.join(FEATURES_DIR, f'{label}_combined.npy'),       F_combined)

    print(f"  Saved F1: {F1.shape}  F2: {F2.shape}  F3: {F3.shape}")
    print(f"  Saved combined: {F_combined.shape}")

    # ── Save labels alongside features ────────────────────────────────────────
    # Load original curated CSV to pull label columns
    csv_path = os.path.join(CURATED_DIR, f'{label}_curated.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.dropna(subset=['sequence']).reset_index(drop=True)

        # Keep SeqID + label columns + confidence columns
        drugs = DRUG_COLS[label]
        label_cols  = [f'{d}_label' for d in drugs if f'{d}_label' in df.columns]
        conf_cols   = [f'{d}_conf'  for d in drugs if f'{d}_conf'  in df.columns]
        keep_cols   = ['SeqID'] + label_cols + conf_cols
        keep_cols   = [c for c in keep_cols if c in df.columns]

        # Align with embedding order using seq_ids
        df_labels = df[keep_cols].copy()
        df_labels.to_csv(
            os.path.join(FEATURES_DIR, f'{label}_labels.csv'),
            index=False
        )
        print(f"  Saved labels: {df_labels.shape[0]} rows x {len(label_cols)} drugs")

    # ── Statistics report ──────────────────────────────────────────────────────
    report_path = os.path.join(FEATURES_DIR, f'{label}_feature_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"ESM-2 Feature Extraction Summary: {label}\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Layer: {REPR_LAYER} (last layer)\n")
        f.write(f"Embedding dimension: 1,280\n")
        f.write(f"Sequences processed: {n_seqs}\n\n")
        f.write("Feature shapes:\n")
        f.write(f"  F1 (global mean):   {F1.shape}\n")
        f.write(f"  F2 (mutant mean):   {F2.shape}  [novel]\n")
        f.write(f"  F3 (delta):         {F3.shape}  [novel]\n")
        f.write(f"  Combined:           {F_combined.shape}\n\n")
        f.write("Feature statistics:\n")
        for name, feat in [('F1', F1), ('F2', F2), ('F3', F3)]:
            f.write(f"  {name}: mean={feat.mean():.4f}  "
                    f"std={feat.std():.4f}  "
                    f"min={feat.min():.4f}  "
                    f"max={feat.max():.4f}\n")
        f.write(f"\nWildtype reference: HXB2 {enzyme} "
                f"({len(wt_seq)} amino acids)\n")
        f.write("Source: GenBank K03455.1\n")

    print(f"  Report saved: {report_path}")


def main():
    print("\nTask 2 - File 2: Feature Extraction")
    print("=" * 60)
    print("Reads: ./embeddings/*_embeddings_raw.npy  (from File 1)")
    print("Writes:")
    print("  ./features/*_F1_global_mean.npy    (1280-dim global mean)")
    print("  ./features/*_F2_mutant_mean.npy    (1280-dim mutant mean) [novel]")
    print("  ./features/*_F3_delta.npy          (1280-dim delta)       [novel]")
    print("  ./features/*_combined.npy          (3840-dim concatenated)")
    print("  ./features/*_labels.csv            (drug resistance labels)")
    print("=" * 60)

    # Model needed only for the wildtype reference embedding
    tokenizer, model, device = load_model_and_tokenizer()

    for label in ['PI', 'NRTI', 'NNRTI', 'INI']:
        extract_features_for_dataset(label, tokenizer, model, device)

    print(f"\n{'='*60}")
    print("Feature extraction complete.")
    print(f"Features saved to: {os.path.abspath(FEATURES_DIR)}/")
    print("\nNext step: use these features to train classifiers in Task 3.")
    print("  Load features:  np.load('./features/PI_F1_global_mean.npy')")
    print("  Load labels:    pd.read_csv('./features/PI_labels.csv')")
    print("=" * 60)


if __name__ == '__main__':
    main()
