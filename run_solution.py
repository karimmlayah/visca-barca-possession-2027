"""
run_solution.py - Reproducible execution entry-point for More Than A Hack 2027.
Generates the validated full-match possession submission CSV.
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.features import extract_features_from_atd, N_VIDEO_FRAMES
from src.priors import load_smart_tagging_priors
from src.model import predict_match_possession

def resolve_data_paths(data_dir=None):
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    if os.environ.get("MTAH_DATA_DIR"):
        candidates.append(os.environ["MTAH_DATA_DIR"])
    candidates.extend([
        os.path.join(HERE, "data"),
        os.path.join(HERE, "..", "data"),
        os.path.join(HERE, "..", "repo", "data"),
        r"D:\downloads\hackathon_barca\data"
    ])
    
    for c in candidates:
        tactical = os.path.join(c, "tactical") if os.path.basename(c) != "tactical" else c
        raw_p = os.path.join(tactical, "BAR-ATH_tactical_atd.txt")
        meta_p = os.path.join(tactical, "BAR-ATH_tactical_atd.xml")
        tag_p = os.path.join(tactical, "BAR-ATH_tactical_smart_tagging.xml")
        if os.path.isfile(raw_p) and os.path.isfile(meta_p) and os.path.isfile(tag_p):
            return raw_p, meta_p, tag_p
            
    raise FileNotFoundError(
        "Could not locate tactical dataset files. Please provide --data-dir or set MTAH_DATA_DIR."
    )

def main():
    parser = argparse.ArgumentParser(description="Reproduce More Than A Hack 2027 Submission")
    parser.add_argument("--data-dir", default=None, help="Root directory of dataset")
    parser.add_argument("--output", default=None, help="Output submission CSV path")
    parser.add_argument("--evaluate", action="store_true", help="Score against public ground truth if available")
    args = parser.parse_args()
    
    t0 = time.time()
    raw_p, meta_p, tag_p = resolve_data_paths(args.data_dir)
    print(f"Using tactical dataset from: {os.path.dirname(raw_p)}")
    
    # 1. Feature extraction
    print("[1/4] Extracting ATD kinematic features...")
    feats = extract_features_from_atd(raw_p, meta_p)
    print(f"      Ball visible on {int(np.sum(feats['ball_seen'])):,} frames")
    
    # 2. Smart Tagging priors
    print("[2/4] Loading tactical priors from Smart Tagging...")
    smart_h, smart_a, sp_dead = load_smart_tagging_priors(tag_p)
    
    # 3. Model inference
    print("[3/4] Running HMM sequence decoding and duration modeling...")
    labels = predict_match_possession(feats, smart_h, smart_a, sp_dead)
    
    # 4. Write submission CSV
    output_path = args.output
    if not output_path:
        sub_dir = os.path.join(HERE, "..", "submission")
        os.makedirs(sub_dir, exist_ok=True)
        output_path = os.path.join(sub_dir, "Visca Barca.csv")
    else:
        out_folder = os.path.dirname(output_path)
        if out_folder:
            os.makedirs(out_folder, exist_ok=True)
            
    print(f"[4/4] Writing submission file to {output_path}...")
    df = pd.DataFrame({
        'frame': np.arange(N_VIDEO_FRAMES, dtype=np.int32),
        'label': labels
    })
    df.to_csv(output_path, index=False)
    
    # Validation checks
    assert len(df) == N_VIDEO_FRAMES, f"Invalid row count: {len(df)}"
    assert list(df.columns) == ['frame', 'label'], f"Invalid columns: {list(df.columns)}"
    assert set(df['label'].unique()).issubset({'H', 'A', 'D', 'X'}), f"Invalid labels: {df['label'].unique()}"
    print("      Verification: PASSED (exactly 199,319 rows, monotonic frames 0..199318, labels H/A/D/X)")
    
    counts = df['label'].value_counts().to_dict()
    print("      Class distribution:", {k: f"{counts.get(k, 0):,} ({counts.get(k, 0)/N_VIDEO_FRAMES*100:.1f}%)" for k in ['H', 'A', 'D', 'X']})
    
    # Optional evaluation
    if args.evaluate:
        repo_eval = os.path.join(HERE, "..", "repo", "evaluation")
        gt_csv = os.path.join(repo_eval, "gt", "BAR-ATH_possession_gt_public.csv")
        if os.path.isfile(gt_csv):
            sys.path.insert(0, os.path.join(HERE, "..", "repo"))
            from evaluation import possession_metrics as pm
            gt_df = pm.read_label_csv(gt_csv)
            gt_codes = pm.label_codes(gt_df)
            pred_codes = pm.label_codes(df)
            cm = pm.confusion(gt_codes, pred_codes)
            cm3 = pm.scored_confusion(cm)
            runs = pm.valid_runs(gt_codes)
            filled = pm.fill_abstentions(pred_codes, runs)
            m_f1 = pm.macro_f1(cm3)
            sp = pm.spell_scores(gt_codes, filled, runs)
            print(f"\n[Public GT Sanity Check] macroF1: {m_f1:.4f} | spellF1: {sp['mean_f1']:.4f}")
            
    print(f"\nPipeline completed successfully in {time.time() - t0:.2f} seconds.")

if __name__ == '__main__':
    main()
