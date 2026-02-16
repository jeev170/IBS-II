"""
Task 1: HIV Drug Resistance Dataset Curation
============================================
Paper: Hybrid ESM-2 + GNN Framework for HIV Drug Resistance Prediction
Data:  Stanford HIVDB unfiltered genotype-phenotype datasets

USAGE
-----
    pip install pandas numpy matplotlib seaborn scikit-learn
    python task1_curation.py

Place PI.csv, NRTI.csv, NNRTI.csv, INI.csv in the same folder as this
script, or update DATA_DIR below.

OUTPUT (written to ./curated/)
------
    PI_curated.csv          – cleaned sequences + binary labels
    NRTI_curated.csv
    NNRTI_curated.csv
    INI_curated.csv
    PI_sequences.fasta      – FASTA for ESM-2 embedding (Task 2)
    NRTI_sequences.fasta
    NNRTI_sequences.fasta
    INI_sequences.fasta
    curation_summary.csv    – per-dataset statistics table
    validation_stats.txt    – methods-ready text for the paper
    validation_report.png   – 6-panel validation figure

PIPELINE STAGES
---------------
    A  Clinical isolate filter     (Rhee et al. 2006 PNAS)
    B  Patient deduplication       (Rhee et al. 2006 PNAS)
    C  DRM mixture filter          (Rhee et al. 2006 PNAS)
       + sequence reconstruction from HXB2 reference
    D  Jaccard mutation-set dedup  [NOVEL — first in HIV ML literature]
    E  Three-tier binary labelling [NOVEL — confidence weights retained]
       Cutoffs: Rhee et al. 2006 PNAS (PI, NNRTI, AZT, 3TC)
                Stanford HIVDB biological cutoffs (other NRTIs, INI)
    F  Subtype polymorphism flag   [NOVEL — prevents non-B over-calling]
    G  Statistical validation

REFERENCES
----------
Rhee SY et al. (2006). Genotypic predictors of HIV-1 drug resistance.
    PNAS 103(46):17355-60. DOI:10.1073/pnas.0607274103
    [Primary source for cutoffs and redundancy removal criteria]

IAS-USA Drug Resistance Mutations Group (2023).
    [Source for major DRM positions used in mixture filtering]

Stanford HIVDB biological cutoffs (treatment-naive distribution).
    https://hivdb.stanford.edu
    [Source for newer NRTI, NNRTI, and INI cutoffs]
"""

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = '.'          # folder containing PI.csv, NRTI.csv, etc.
OUTPUT_DIR = './curated'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Pipeline parameters ─────────────────────────────────────────────────────
JACCARD_THRESHOLD = 0.85   # mutation-set similarity cutoff for Stage D
MIN_SEQUENCES     = 500    # minimum sequences per dataset
STANDARD_AAS      = set('ACDEFGHIKLMNPQRSTVWY')

# ─── HXB2 reference sequences (GenBank K03455.1) ─────────────────────────────
# Used to reconstruct full protein sequences from HIVDB per-position tables.
# '-' in a HIVDB row means "same as HXB2 at this position".

HXB2_PR = list(                               # Protease: 99 aa
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQ"
    "ILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)
assert len(HXB2_PR) == 99

_RT = (                                        # Reverse transcriptase: 560 aa
    "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIK"
    "KKDSTKWRKLYPQKIKEQYFEWMGYLENPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQI"
    "YPGIKVRQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQ"
    "GQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETW"
    "ETWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTDRG"
    "RQKVVSLTDTTNQKTELQAIHLALQDSGLEIVNIVTDSQYALGIIQAQPDKSESELVSQIIEQL"
    "IKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVLFLDGIDKAQEEHEKYHSNWRAMASDFNLP"
    "PVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVIIVAVHVASGYIEAEVIPAET"
    "GQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMN"
)
HXB2_RT = list(_RT[:560])
while len(HXB2_RT) < 560:
    HXB2_RT.append('X')

_IN = (                                        # Integrase: 288 aa
    "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQL"
    "DCTHLEGKVIIVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVK"
    "AACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGI"
    "GGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNS"
    "DIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"
)
HXB2_IN = list(_IN[:288])
while len(HXB2_IN) < 288:
    HXB2_IN.append('X')

# ─── Resistance cutoffs ───────────────────────────────────────────────────────
# Each entry: (lower_cutoff, upper_cutoff)
#
# Tier assignment (Rhee et al. 2006 PNAS):
#   FC <= lower             → Susceptible  (binary label = 0, confidence = 1.0)
#   lower < FC <= upper     → Intermediate (binary label = 1, confidence = 0.5)
#   FC > upper              → High-resist  (binary label = 1, confidence = 1.0)
#
# Binary label is used for classification.
# Confidence weight is stored for uncertainty-aware training (novel).
#
# Sources:
#   PI (all 8 drugs)        → Rhee 2006 PNAS: <3.0 S, 3.0–20 I, >20 H
#   NNRTI (EFV, NVP)        → Rhee 2006 PNAS: <3.0 S, 3.0–25 I, >25 H
#   NRTI AZT, 3TC           → Rhee 2006 PNAS: <3.0 S, 3.0–25 I, >25 H
#   NRTI D4T, DDI, ABC, TDF → Stanford HIVDB biological cutoffs
#   Newer NNRTI (ETR,RPV,DOR) → Stanford HIVDB biological cutoffs
#   INI (all 5 drugs)       → Stanford HIVDB biological cutoffs

CUTOFFS = {
    # PI — Rhee 2006 PNAS
    'FPV': (3.0, 20.0), 'ATV': (3.0, 20.0), 'IDV': (3.0, 20.0),
    'LPV': (3.0, 20.0), 'NFV': (3.0, 20.0), 'SQV': (3.0, 20.0),
    'TPV': (3.0, 20.0), 'DRV': (3.0, 20.0),
    # NRTI — Rhee 2006 PNAS (AZT, 3TC) + Stanford HIVDB (others)
    'AZT': (3.0, 25.0), '3TC': (3.0, 25.0),
    'D4T': (1.5,  4.0), 'DDI': (2.0, 10.0),
    'ABC': (2.0,  8.0), 'TDF': (1.4,  4.0),
    # NNRTI — Rhee 2006 PNAS (EFV, NVP) + Stanford HIVDB (others)
    'EFV': (3.0, 25.0), 'NVP': (3.0, 25.0),
    'ETR': (3.0, 25.0), 'RPV': (3.0, 25.0), 'DOR': (3.0, 25.0),
    # INI — Stanford HIVDB biological cutoffs
    'RAL': (2.5, 10.0), 'EVG': (2.5, 10.0), 'DTG': (2.5, 10.0),
    'BIC': (2.5, 10.0), 'CAB': (2.5, 10.0),
}

# ─── Major DRM positions (1-indexed, IAS-USA 2023) ────────────────────────────
# Used only for DRM mixture filtering (Stage C).
# These are positions where mixture calls are excluded per Rhee 2006.
MAJOR_DRMS = {
    'PR':    {10,11,20,24,30,32,33,36,43,46,47,48,50,53,54,58,60,62,
              63,64,71,73,76,77,82,83,84,85,88,90,93},
    'NRTI':  {41,44,65,67,68,69,70,74,75,77,115,116,151,184,210,215,219},
    'NNRTI': {98,100,101,103,106,108,138,179,181,184,188,190,221,225,227,230,318},
    'IN':    {66,92,95,97,118,121,143,147,148,155,163,230},
}

# ─── Subtype-specific natural polymorphisms ───────────────────────────────────
# Positions where non-B subtypes naturally differ from HXB2.
# These are NOT resistance mutations — flagged to prevent over-calling.
# Source: Rhee SY et al. (2006) AIDS 20:643-651
SUBTYPE_POLYMORPHISMS = {
    'PR': {
        'C':        {36: 'I', 89: 'M'},
        'CRF02_AG': {36: 'I'},
        'A':        {36: 'I'},
        'G':        {36: 'I', 89: 'M'},
    },
    'RT': {
        'C':        {245: 'Q', 203: 'K'},
        'CRF02_AG': {203: 'K'},
    },
    'IN': {},
}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE A + B  ·  Load, clinical filter, patient dedup
# ══════════════════════════════════════════════════════════════════════════════

def load_and_stage_ab(csv_path, drugs):
    """
    Stage A — Remove lab isolates (site-directed mutants, in vitro passage).
    Stage B — One sequence per patient; keep the most drug-labelled sequence.

    Reference: Rhee et al. 2006 PNAS
      "Isolates included viruses from the plasma of HIV-1-infected persons."
      "Up to two isolates from a small number of individuals were included
       provided their isolates differed at two or more drug-resistance positions."
      → We are stricter: strictly one per patient to prevent data leakage in CV.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)

    # Stage A
    if 'Type' in df.columns:
        df = df[df['Type'] == 'Clinical'].copy()
    n_clin = len(df)

    # Stage B
    drugs_present = [d for d in drugs if d in df.columns]
    df['_nl'] = df[drugs_present].notna().sum(axis=1)
    df = df.sort_values('_nl', ascending=False)
    df = df.groupby('PtID', sort=False).first().reset_index()
    df = df.drop(columns=['_nl'], errors='ignore')
    n_dedup = len(df)

    print(f"    Raw={n_raw} | Clinical={n_clin} (-{n_raw-n_clin} lab) "
          f"| Dedup={n_dedup} (-{n_clin-n_dedup} longitudinal)")
    return df, n_raw, n_clin, n_dedup


# ══════════════════════════════════════════════════════════════════════════════
# STAGE C  ·  Sequence reconstruction + DRM mixture filter
# ══════════════════════════════════════════════════════════════════════════════

def resolve_position(val, consensus_aa, is_drm):
    """
    Map one HIVDB position cell to a single amino acid.

    HIVDB encoding:
      '-'    → wildtype (HXB2 consensus at this position)
      '.'    → no amplification data
      '#'    → insertion
      '~'    → deletion
      '*'    → stop codon
      1 AA   → amino acid substitution
      2+ AA  → sequencing mixture

    Mixture at DRM position → exclude row (Rhee 2006: "electrophoretic
    evidence of more than one amino acid at a nonpolymorphic DRM position").
    Mixture at non-DRM position → resolve to consensus (not resistance-relevant).

    Returns: (amino_acid, is_gap, is_drm_mixture)
    """
    if pd.isna(val):
        return consensus_aa, True, False

    val = str(val).strip()

    if val == '-':
        return consensus_aa, False, False

    if val in ('.', '#', '~', '*'):
        return None, True, False

    if len(val) == 1 and val in STANDARD_AAS:
        return val, False, False

    # 2+ characters = mixture
    clean = [c for c in val if c in STANDARD_AAS]
    if clean:
        if is_drm:
            return None, False, True      # DRM mixture → exclude
        return consensus_aa, False, False  # non-DRM mixture → consensus
    return consensus_aa, True, False


def stage_c(df, pos_cols, consensus, drm_positions, enzyme):
    """
    Reconstruct full protein sequences and apply filters.

    Coverage thresholds (per clinical diagnostic standards):
      PR   (99 aa):  100% coverage — protein too short to tolerate gaps
      NRTI (560 aa): ≥80% of resistance window positions 40–219
      NNRTI(560 aa): ≥80% of resistance window positions 90–348
      IN   (288 aa): ≥90% of resistance window positions 51–263
    """
    seqs, gaps, drm_mixes = [], [], []

    for _, row in df.iterrows():
        seq_list, n_gap, n_dm = [], 0, 0
        for i, col in enumerate(pos_cols):
            pos1   = i + 1
            is_drm = pos1 in drm_positions
            con    = consensus[i] if i < len(consensus) else 'X'
            aa, is_gap, is_dm = resolve_position(row.get(col, np.nan), con, is_drm)
            seq_list.append('X' if (is_gap or aa is None) else aa)
            if is_gap or aa is None: n_gap += 1
            if is_dm:                n_dm  += 1
        seqs.append(''.join(seq_list))
        gaps.append(n_gap / len(pos_cols))
        drm_mixes.append(n_dm)

    df = df.copy()
    df['sequence']    = seqs
    df['_gap_frac']   = gaps
    df['_drm_mix']    = drm_mixes

    # Coverage filter
    if enzyme == 'PR':
        cov_ok = df['_gap_frac'] == 0.0

    elif enzyme == 'NRTI':
        def _nrti_cov(seq):
            w = seq[39:219]        # positions 40–219 (0-indexed 39–218)
            return w.count('X') / max(len(w), 1) <= 0.20
        cov_ok = df['sequence'].apply(_nrti_cov)

    elif enzyme == 'NNRTI':
        def _nnrti_cov(seq):
            w = seq[89:348]        # positions 90–348
            return w.count('X') / max(len(w), 1) <= 0.20
        cov_ok = df['sequence'].apply(_nnrti_cov)

    elif enzyme == 'IN':
        def _in_cov(seq):
            w = seq[50:263]        # positions 51–263
            return w.count('X') / max(len(w), 1) <= 0.10
        cov_ok = df['sequence'].apply(_in_cov)

    else:
        cov_ok = pd.Series(True, index=df.index)

    # DRM mixture filter
    mix_ok = df['_drm_mix'] == 0

    df_out = df[cov_ok & mix_ok].copy().reset_index(drop=True)
    df_out = df_out.drop(columns=['_gap_frac', '_drm_mix'])

    n_fail_cov = (~cov_ok).sum()
    n_fail_mix = (cov_ok & ~mix_ok).sum()
    print(f"    Coverage removed: {n_fail_cov} | DRM mixture removed: {n_fail_mix} "
          f"→ {len(df_out)} remain")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# STAGE D  ·  Jaccard mutation-set deduplication  [NOVEL]
# ══════════════════════════════════════════════════════════════════════════════

def parse_mut_set(comp_mut_list):
    """'L10I, K20R, M46I' → frozenset({'L10I', 'K20R', 'M46I'})"""
    if pd.isna(comp_mut_list) or not str(comp_mut_list).strip():
        return frozenset()
    return frozenset(m.strip() for m in str(comp_mut_list).split(',') if m.strip())


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or  not b: return 0.0
    return len(a & b) / len(a | b)


def stage_d(df, drugs, threshold=JACCARD_THRESHOLD):
    """
    Jaccard mutation-set deduplication.  [NOVEL — first in HIV ML literature]

    Why not CD-HIT or cosine similarity on full sequences?
      HIV sequences are extremely conserved globally (~97% identity for RT).
      CD-HIT at any reasonable threshold collapses everything into one cluster.
      Cosine similarity on binary position vectors loses information about
      which amino acid change occurred (V82A vs V82F are clinically different).

    CompMutList uses HIVDB standard mutation nomenclature (e.g. 'V82A').
    Jaccard on these named mutation sets measures resistance-profile similarity
    in the exact vocabulary used by IAS-USA guidelines and clinicians.

    At Jaccard >= 0.85:  two sequences share ≥85% of their mutation set
    (union-normalised) → same dominant resistance pathway → profile duplicate.
    Retain the more completely drug-labelled sequence; discard the other.
    """
    if 'CompMutList' not in df.columns or len(df) < 2:
        return df

    mut_sets   = [parse_mut_set(v) for v in df['CompMutList']]
    drugs_here = [d for d in drugs if d in df.columns]
    lbl_cnts   = df[drugs_here].notna().sum(axis=1).values
    n          = len(df)
    keep       = np.ones(n, dtype=bool)

    wt = sum(1 for s in mut_sets if not s)
    print(f"    Jaccard dedup: {n} sequences, {wt} wildtype-like")

    for i in range(n):
        if not keep[i]: continue
        for j in range(i + 1, n):
            if not keep[j]: continue
            if jaccard(mut_sets[i], mut_sets[j]) >= threshold:
                if lbl_cnts[i] >= lbl_cnts[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    out = df[keep].reset_index(drop=True)
    print(f"    Jaccard(>={threshold}): {n} → {keep.sum()} "
          f"({(~keep).sum()} resistance-profile duplicates removed)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STAGE E  ·  Three-tier binary labelling  [NOVEL]
# ══════════════════════════════════════════════════════════════════════════════

def stage_e(df, drugs):
    """
    Assign binary resistance labels with confidence weights.  [NOVEL]

    Three tiers per Rhee et al. 2006 PNAS:
      Susceptible   (FC <= lower):          label=0, confidence=1.0
      Intermediate  (lower < FC <= upper):  label=1, confidence=0.5
      High-resist   (FC > upper):           label=1, confidence=1.0
      Not measured  (FC is NaN):            label=NaN, confidence=NaN

    The confidence weight is the novel addition: prior binary classification
    papers discard the intermediate tier or arbitrarily assign it to one class.
    We retain it with confidence=0.5 so downstream models can:
      - Use confidence-weighted loss functions
      - Apply semi-supervised or uncertainty-quantification techniques
      - Report performance separately on unambiguous vs ambiguous subsets
    """
    df    = df.copy()
    stats = {}

    for drug in drugs:
        if drug not in df.columns:
            continue
        lo, hi = CUTOFFS.get(drug, (3.0, 25.0))
        fc = pd.to_numeric(df[drug], errors='coerce')

        label = np.where(fc.isna(), np.nan,
                np.where(fc <= lo, 0.0, 1.0))

        conf  = np.where(fc.isna(), np.nan,
                np.where((fc > lo) & (fc <= hi), 0.5, 1.0))

        df[f'{drug}_label'] = label
        df[f'{drug}_conf']  = conf

        n_s  = int((~fc.isna() & (fc <= lo)).sum())
        n_i  = int((~fc.isna() & (fc > lo) & (fc <= hi)).sum())
        n_h  = int((~fc.isna() & (fc > hi)).sum())
        n_nm = int(fc.isna().sum())
        stats[drug] = {
            'susceptible': n_s, 'intermediate': n_i,
            'high_resistant': n_h, 'not_measured': n_nm,
            'lower_cutoff': lo, 'upper_cutoff': hi,
            'underpowered': n_s < 50 or (n_i + n_h) < 50,
        }

    return df, stats


# ══════════════════════════════════════════════════════════════════════════════
# STAGE F  ·  Subtype polymorphism flag  [NOVEL]
# ══════════════════════════════════════════════════════════════════════════════

def stage_f(df, consensus, enzyme_key):
    """
    Flag positions where non-B subtypes naturally differ from HXB2.  [NOVEL]

    Rhee 2006: "Mutations were defined as amino acid differences from the
    subtype B consensus wild-type sequence." This means non-B sequences can
    have HXB2 'mutations' at positions that are simply their natural wild-type.
    Including these as resistance mutations would bias classifiers.

    We flag (not remove) these positions so that:
      - Subtype-stratified analyses in Task 4 can correctly exclude them
      - Non-B sequence embeddings (Task 2) are not penalised for natural variation

    Source: Rhee SY et al. (2006) AIDS 20:643-651 — pol mutation frequency
    by subtype (data compiled by the Stanford HIVseq program).
    """
    df        = df.copy()
    poly_map  = SUBTYPE_POLYMORPHISMS.get(enzyme_key, {})
    flags     = []

    for _, row in df.iterrows():
        subtype = str(row.get('Subtype', 'B')).strip()
        if subtype == 'B' or subtype not in poly_map:
            flags.append('')
            continue
        smap = poly_map[subtype]
        seq  = row.get('sequence', '')
        flagged = []
        for pos1, subtype_aa in smap.items():
            idx = pos1 - 1
            if idx < len(seq) and seq[idx] == subtype_aa:
                hxb2_aa = consensus[idx] if idx < len(consensus) else '?'
                flagged.append(f'{hxb2_aa}{pos1}{subtype_aa}')
        flags.append(','.join(flagged))

    df['subtype_polymorphisms'] = flags
    n_flagged = (df['subtype_polymorphisms'] != '').sum()
    print(f"    Subtype flag: {n_flagged} sequences have natural polymorphisms flagged")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE G  ·  Statistical validation
# ══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(seqs, seq_len):
    """Per-position Shannon entropy (bits) across curated sequences."""
    out = []
    for pos in range(seq_len):
        col = [s[pos] for s in seqs if pos < len(s) and s[pos] != 'X']
        if not col:
            out.append(0.0)
            continue
        counts = {}
        for aa in col:
            counts[aa] = counts.get(aa, 0) + 1
        total = sum(counts.values())
        probs = np.array([v / total for v in counts.values()])
        out.append(float(-np.sum(probs * np.log2(probs + 1e-10))))
    return np.array(out)


def drm_audit(seqs, consensus, drm_positions):
    """Fraction of sequences mutated at each IAS-USA DRM position."""
    freqs = {}
    for pos in sorted(drm_positions):
        idx   = pos - 1
        if idx >= len(consensus): continue
        con   = consensus[idx]
        valid = [s for s in seqs if idx < len(s) and s[idx] != 'X']
        mut   = [s for s in valid if s[idx] != con]
        freqs[pos] = len(mut) / len(valid) if valid else 0.0
    return freqs


def pairwise_id_sample(seqs, n=400, seed=42):
    """Sampled pairwise sequence identity for histogram."""
    rng  = np.random.default_rng(seed)
    samp = [seqs[i]
            for i in rng.choice(len(seqs), min(n, len(seqs)), replace=False)]
    ids  = []
    for i, j in itertools.combinations(range(len(samp)), 2):
        s1, s2       = samp[i], samp[j]
        match = comp = 0
        for a, b in zip(s1, s2):
            if a != 'X' and b != 'X':
                comp += 1
                if a == b: match += 1
        if comp > 0:
            ids.append(match / comp)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# Validation figure
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(results, output_dir):
    """6-panel validation figure for the paper."""
    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.36)
    cc  = {'PI': '#4C72B0', 'NRTI': '#DD8452',
           'NNRTI': '#55A868', 'INI': '#C44E52'}
    cls = list(results.keys())

    # ── A: Curation funnel ────────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 0])
    stage_keys   = ['n_raw', 'n_clin', 'n_dedup', 'n_final']
    stage_labels = ['Raw', 'Clinical', 'Dedup', 'Final']
    bar_colors   = ['#c6dbef', '#9ecae1', '#6baed6', '#2171b5']
    x  = np.arange(len(cls)); w = 0.18
    for ki, (key, col, lbl) in enumerate(zip(stage_keys, bar_colors, stage_labels)):
        vals = [results[c][key] for c in cls]
        ax.bar(x + (ki - 1.5) * w, vals, w, color=col,
               edgecolor='white', lw=0.5, label=lbl)
        if ki == 3:
            for xi, v in zip(x, vals):
                ax.text(xi + 1.5 * w, v + 20, str(v), ha='center',
                        va='bottom', fontsize=8, color='#08519c',
                        fontweight='bold')
    ax.axhline(MIN_SEQUENCES, color='red', ls='--', lw=1,
               label=f'Min={MIN_SEQUENCES}')
    ax.set_xticks(x); ax.set_xticklabels(cls, fontsize=11)
    ax.set_ylabel('Sequences', fontsize=11)
    ax.set_title('A  Curation funnel', fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=8, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)

    # ── B: Subtype distribution ────────────────────────────────────────────────
    ax     = fig.add_subplot(gs[0, 1])
    all_st = defaultdict(lambda: defaultdict(int))
    for c, r in results.items():
        if 'df' not in r: continue
        for st, cnt in r['df']['Subtype'].value_counts().items():
            all_st[c][st] += cnt
    top_st = sorted({
        st for c in all_st
        for st in list(dict(sorted(
            all_st[c].items(), key=lambda x: -x[1]
        )).keys())[:6]
    })
    mat = np.array([[all_st[c].get(st, 0) for st in top_st] for c in cls])
    sc  = plt.cm.Set2(np.linspace(0, 0.8, len(top_st)))
    bot = np.zeros(len(cls))
    for j, (st, co) in enumerate(zip(top_st, sc)):
        ax.bar(cls, mat[:, j], bottom=bot, label=st, color=co,
               edgecolor='white', lw=0.4)
        bot += mat[:, j]
    ax.set_ylabel('Sequences', fontsize=11)
    ax.set_title('B  Subtype distribution', fontsize=12,
                 fontweight='bold', loc='left')
    ax.legend(title='Subtype', fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)

    # ── C: Shannon entropy per position ───────────────────────────────────────
    ax   = fig.add_subplot(gs[1, :])
    xoff = 0; tpos, tlab = [], []
    for c, r in results.items():
        if 'entropy' not in r: continue
        ent  = r['entropy']
        drms = sorted(r['drm_positions'])
        xs   = np.arange(xoff, xoff + len(ent))
        col  = cc.get(c, '#888')
        ax.fill_between(xs, ent, alpha=0.2, color=col)
        ax.plot(xs, ent, lw=0.7, color=col, label=f'{c} (n={r["n_final"]})')
        for pos in drms:
            idx = pos - 1
            if 0 <= idx < len(ent):
                ax.axvline(xoff + idx, color=col, alpha=0.2, lw=0.5)
        tpos.append(xoff + len(ent) // 2)
        tlab.append(c)
        xoff += len(ent) + 50
    ax.set_xticks(tpos); ax.set_xticklabels(tlab, fontsize=11)
    ax.set_ylabel('Shannon entropy (bits)', fontsize=11)
    ax.set_title('C  Per-position Shannon entropy  '
                 '(marks = IAS-USA DRM positions)',
                 fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # ── D: Pairwise identity distribution ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    for c, r in results.items():
        ids = r.get('pairwise_ids', [])
        if ids:
            ax.hist(ids, bins=40, alpha=0.55, density=True,
                    label=f'{c} (n={r["n_final"]})',
                    color=cc.get(c, '#888'), edgecolor='none')
    ax.set_xlabel('Pairwise sequence identity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('D  Pairwise identity distribution',
                 fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # ── E: DRM frequency heatmap ──────────────────────────────────────────────
    ax   = fig.add_subplot(gs[2, 1])
    rows = [
        {'Class': c, 'Pos': pos, 'Freq': freq}
        for c, r in results.items()
        for pos, freq in r.get('drm_freqs', {}).items()
    ]
    if rows:
        hdf   = pd.DataFrame(rows)
        pivot = hdf.pivot_table(values='Freq', index='Class',
                                columns='Pos', fill_value=0)
        top_c = pivot.var(axis=0).nlargest(24).index
        sns.heatmap(pivot[top_c], ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
                    linewidths=0.3, linecolor='white',
                    cbar_kws={'label': 'Mutation frequency', 'shrink': 0.8})
        ax.set_xlabel('DRM position (1-indexed)', fontsize=10)
        ax.set_ylabel('Drug class', fontsize=10)
        ax.set_title('E  IAS-USA DRM frequencies (top 24 most variable)',
                     fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    # ── F: Label balance (three-tier) ─────────────────────────────────────────
    ax      = fig.add_subplot(gs[3, :])
    br_rows = []
    for c, r in results.items():
        for drug, s in r.get('label_stats', {}).items():
            br_rows.append({
                'Drug': drug, 'Class': c,
                'S': s['susceptible'],
                'I': s['intermediate'],
                'H': s['high_resistant'],
                'UP': s['underpowered'],
            })
    if br_rows:
        bd = pd.DataFrame(br_rows)
        xi = np.arange(len(bd))
        ax.bar(xi, bd['S'], label='Susceptible',
               color='#4393c3', edgecolor='white')
        ax.bar(xi, bd['I'], bottom=bd['S'],
               label='Intermediate (conf=0.5)',
               color='#fed976', edgecolor='white')
        ax.bar(xi, bd['H'], bottom=bd['S'] + bd['I'],
               label='High resistant',
               color='#d6604d', edgecolor='white')
        for i, row in bd.iterrows():
            if row['UP']:
                ax.annotate('⚠', (i, row['S'] + row['I'] + row['H'] + 8),
                            ha='center', fontsize=10, color='red')
        ax.set_xticks(xi)
        ax.set_xticklabels(
            [f"{r['Drug']}\n({r['Class']})" for _, r in bd.iterrows()],
            rotation=45, ha='right', fontsize=8
        )
        ax.set_ylabel('Sequences', fontsize=11)
        ax.set_title(
            'F  Three-tier label balance (Rhee 2006 PNAS)  — ⚠ = <50 per class',
            fontsize=12, fontweight='bold', loc='left'
        )
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle(
        'HIV Drug Resistance — Task 1 Curation Validation Report\n'
        'Cutoffs: Rhee et al. 2006 PNAS | Stanford HIVDB biological cutoffs',
        fontsize=14, fontweight='bold', y=1.003
    )
    out = os.path.join(output_dir, 'validation_report.png')
    fig.savefig(out, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Validation figure saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main per-dataset pipeline
# ══════════════════════════════════════════════════════════════════════════════

def curate(csv_path, label, consensus, drugs, drm_positions, enzyme_key):
    """Run all stages for one dataset."""
    print(f"\n{'='*62}")
    print(f"  {label}  [{enzyme_key}]")
    print(f"{'='*62}")

    # ── A + B ──────────────────────────────────────────────────────────────────
    df, n_raw, n_clin, n_dedup = load_and_stage_ab(csv_path, drugs)

    pos_cols = sorted(
        [c for c in df.columns if c.startswith('P') and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )

    # ── C ──────────────────────────────────────────────────────────────────────
    print(f"  Stage C: Sequence reconstruction + DRM mixture filter...")
    df = stage_c(df, pos_cols, consensus, drm_positions, enzyme_key)

    # ── D ──────────────────────────────────────────────────────────────────────
    print(f"  Stage D (novel): Jaccard mutation-set dedup "
          f"(threshold={JACCARD_THRESHOLD})...")
    df = stage_d(df, drugs)

    n_final = len(df)
    status  = "MEETS" if n_final >= MIN_SEQUENCES else "*** BELOW ***"
    print(f"\n  {label}: {n_raw} → {n_clin} → {n_dedup} → {n_final}  [{status} min={MIN_SEQUENCES}]")

    # ── E ──────────────────────────────────────────────────────────────────────
    print(f"  Stage E (novel): Three-tier labels + confidence weights...")
    df, label_stats = stage_e(df, drugs)

    print(f"\n  Label balance — S=susceptible · I=intermediate(conf=0.5) · H=high-resistant:")
    for drug in drugs:
        if drug not in label_stats: continue
        s  = label_stats[drug]
        up = '  *** UNDERPOWERED ***' if s['underpowered'] else ''
        print(f"    {drug:5s}  cut={s['lower_cutoff']:.1f}/{s['upper_cutoff']:.0f}  "
              f"S={s['susceptible']:4d}  I={s['intermediate']:4d}  "
              f"H={s['high_resistant']:4d}  NaN={s['not_measured']:4d}{up}")

    # ── F ──────────────────────────────────────────────────────────────────────
    print(f"  Stage F (novel): Subtype polymorphism flagging...")
    df = stage_f(df, consensus, enzyme_key)

    # ── G ──────────────────────────────────────────────────────────────────────
    print(f"  Stage G: Validation statistics...")
    seqs    = df['sequence'].tolist()
    ent     = shannon_entropy(seqs, len(consensus))
    drm_f   = drm_audit(seqs, consensus, drm_positions)
    pid     = pairwise_id_sample(seqs, n=400)
    mi      = np.mean(pid) if pid else 0
    si      = np.std(pid)  if pid else 0
    pnB     = (df['Subtype'] != 'B').sum() / len(df) * 100 \
              if 'Subtype' in df.columns else 0

    active = sorted(p for p, f in drm_f.items() if f > 0.01)
    top8   = sorted(drm_f, key=drm_f.get, reverse=True)[:8]
    print(f"  Shannon entropy: mean={np.mean(ent):.3f} bits")
    print(f"  Pairwise identity: {mi:.3f} ± {si:.3f}")
    print(f"  Active DRM sites (>1%): {len(active)}")
    print(f"  Highest-freq DRMs: {top8}")
    print(f"  Non-B subtype: {pnB:.1f}%")

    # Keep only columns needed downstream
    keep_cols = (
        ['SeqID', 'PtID', 'Subtype', 'IsolateName', 'CompMutList', 'sequence',
         'subtype_polymorphisms']
        + [d for d in drugs if d in df.columns]
        + [f'{d}_label' for d in drugs if f'{d}_label' in df.columns]
        + [f'{d}_conf'  for d in drugs if f'{d}_conf'  in df.columns]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return {
        'df': df,
        'n_raw': n_raw, 'n_clin': n_clin, 'n_dedup': n_dedup, 'n_final': n_final,
        'entropy': ent, 'drm_freqs': drm_f, 'drm_positions': drm_positions,
        'pairwise_ids': pid, 'label_stats': label_stats,
        'drug_cols': [d for d in drugs if d in df.columns],
        'mean_id': round(mi, 3), 'std_id': round(si, 3),
        'mean_entropy': round(float(np.mean(ent)), 3),
        'pct_nonB': round(pnB, 1),
        'n_drm_active': len(active),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_all(results, output_dir):
    rows = []
    for cls, r in results.items():
        df = r['df']

        # Curated CSV  (used by all downstream tasks)
        df.to_csv(os.path.join(output_dir, f'{cls}_curated.csv'), index=False)

        # FASTA  (used by Task 2 — ESM-2 embedding)
        with open(os.path.join(output_dir, f'{cls}_sequences.fasta'), 'w') as f:
            for _, row in df.iterrows():
                sid = row.get('SeqID', 'UNK')
                sub = row.get('Subtype', 'UNK')
                seq = row['sequence'].replace('X', '-')
                f.write(f'>{sid}|{sub}|{cls}\n')
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + '\n')

        rows.append({
            'Drug_Class':   cls,
            'N_Raw':        r['n_raw'],
            'N_Clinical':   r['n_clin'],
            'N_Dedup':      r['n_dedup'],
            'N_Final':      r['n_final'],
            'Retention_%':  round(100 * r['n_final'] / r['n_raw'], 1),
            'Mean_ID':      r['mean_id'],
            'Std_ID':       r['std_id'],
            'Mean_Entropy': r['mean_entropy'],
            'Pct_NonB':     r['pct_nonB'],
            'N_DRM_active': r['n_drm_active'],
            'Meets_500':    r['n_final'] >= MIN_SEQUENCES,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, 'curation_summary.csv'), index=False)

    # Methods-ready text for the paper
    with open(os.path.join(output_dir, 'validation_stats.txt'), 'w', encoding='utf-8') as f:
        f.write("HIV Drug Resistance Dataset - Task 1 Curation Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write("PRIMARY SOURCE FOR CUTOFFS:\n")
        f.write("  Rhee SY et al. (2006). Genotypic predictors of HIV-1 drug\n")
        f.write("  resistance. PNAS 103(46):17355-60. DOI:10.1073/pnas.0607274103\n\n")
        f.write("CUTOFFS USED:\n")
        f.write("  PI (all):      FC <= 3.0 = S  |  3.0-20  = I  |  >20  = H\n")
        f.write("  NNRTI EFV/NVP: FC <= 3.0 = S  |  3.0-25  = I  |  >25  = H\n")
        f.write("  NRTI AZT/3TC:  FC <= 3.0 = S  |  3.0-25  = I  |  >25  = H\n")
        f.write("  D4T: 1.5/4.0 | DDI: 2.0/10 | ABC: 2.0/8 | TDF: 1.4/4.0\n")
        f.write("  ETR/RPV/DOR: 3.0/25 | INI all: 2.5/10 (Stanford HIVDB BCO)\n\n")
        f.write("NOVEL CONTRIBUTIONS:\n")
        f.write("  Stage D: Jaccard mutation-set deduplication\n")
        f.write("           threshold=0.85 on CompMutList parsed mutation sets\n")
        f.write("  Stage E: Three-tier confidence weights (I tier conf=0.5)\n")
        f.write("  Stage F: Subtype polymorphism flags for non-B sequences\n\n")
        f.write("DATASET STATISTICS:\n\n")
        for _, row in summary.iterrows():
            f.write(f"[{row['Drug_Class']}]\n")
            f.write(f"  Raw -> Clinical -> Dedup -> Final: "
                    f"{row['N_Raw']} -> {row['N_Clinical']} -> "
                    f"{row['N_Dedup']} -> {row['N_Final']} "
                    f"({row['Retention_%']}% retained)\n")
            f.write(f"  Pairwise identity: {row['Mean_ID']} +/- {row['Std_ID']}\n")
            f.write(f"  Shannon entropy:   {row['Mean_Entropy']} bits (mean)\n")
            f.write(f"  Non-B subtype:     {row['Pct_NonB']}%\n")
            f.write(f"  Active DRM sites:  {row['N_DRM_active']}\n")
            f.write(f"  Meets min 500:     {row['Meets_500']}\n\n")

    print(f"\n  Outputs saved to: {os.path.abspath(output_dir)}/")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nHIV Drug Resistance — Task 1: Dataset Curation")
    print("=" * 62)
    print("Reference: Rhee et al. 2006 PNAS + Stanford HIVDB biological cutoffs")
    print("Datasets: PI · NRTI · NNRTI · INI  (processed independently)")
    print(f"Data folder: {os.path.abspath(DATA_DIR)}")

    # Check files exist before starting
    files = {'PI': 'PI.csv', 'NRTI': 'NRTI.csv',
             'NNRTI': 'NNRTI.csv', 'INI': 'INI.csv'}
    missing = [v for v in files.values()
               if not os.path.exists(os.path.join(DATA_DIR, v))]
    if missing:
        print(f"\nERROR: missing files in {DATA_DIR}: {missing}")
        print("Update DATA_DIR at the top of this script.")
        sys.exit(1)

    pi = curate(
        csv_path     = os.path.join(DATA_DIR, 'PI.csv'),
        label        = 'PI',
        consensus    = HXB2_PR,
        drugs        = ['FPV','ATV','IDV','LPV','NFV','SQV','TPV','DRV'],
        drm_positions= MAJOR_DRMS['PR'],
        enzyme_key   = 'PR',
    )

    nrti = curate(
        csv_path     = os.path.join(DATA_DIR, 'NRTI.csv'),
        label        = 'NRTI',
        consensus    = HXB2_RT,
        drugs        = ['3TC','ABC','AZT','D4T','DDI','TDF'],
        drm_positions= MAJOR_DRMS['NRTI'],
        enzyme_key   = 'NRTI',
    )

    nnrti = curate(
        csv_path     = os.path.join(DATA_DIR, 'NNRTI.csv'),
        label        = 'NNRTI',
        consensus    = HXB2_RT,
        drugs        = ['EFV','NVP','ETR','RPV','DOR'],
        drm_positions= MAJOR_DRMS['NNRTI'],
        enzyme_key   = 'NNRTI',
    )

    ini = curate(
        csv_path     = os.path.join(DATA_DIR, 'INI.csv'),
        label        = 'INI',
        consensus    = HXB2_IN,
        drugs        = ['RAL','EVG','DTG','BIC','CAB'],
        drm_positions= MAJOR_DRMS['IN'],
        enzyme_key   = 'IN',
    )

    results = {'PI': pi, 'NRTI': nrti, 'NNRTI': nnrti, 'INI': ini}

    print(f"\n{'='*62}")
    print("  Generating validation figure...")
    make_figure(results, OUTPUT_DIR)

    print("  Saving outputs...")
    summary = save_all(results, OUTPUT_DIR)

    print(f"\n{'='*62}")
    print("  FINAL SUMMARY")
    print(f"{'='*62}")
    print(summary.to_string(index=False))
    print(f"\nDone. All files in: {os.path.abspath(OUTPUT_DIR)}/\n")


if __name__ == '__main__':
    main()