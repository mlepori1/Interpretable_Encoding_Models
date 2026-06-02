#!/usr/bin/env python3
"""
Lookup which voxels use a specific feature and display their metadata.

Usage:
    python lookup_feature_voxels.py \
        --csv ../results/full_features/Exp1/sae/gemma-2-2b-res-matryoshka-dc/mean/12/raw/Experiment_1/Qualitative_Analysis_per_participant.csv \
        --feature 199 \
        --dataset ghost
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_feature_indices(feature_str: str) -> list[int]:
    """Parse feature_indices string like '[    40    -71    -79]' into list of ints."""
    if pd.isna(feature_str):
        return []
    # Remove brackets and split on whitespace
    clean = feature_str.strip().strip('[]')
    if not clean:
        return []
    # Split on whitespace and convert to integers
    parts = clean.split()
    return [int(x) for x in parts if x]


def main():
    parser = argparse.ArgumentParser(
        description='Find voxels that use a specific feature and display their metadata.'
    )
    parser.add_argument(
        '--csv', required=True,
        help='Path to Qualitative_Analysis_per_participant.csv'
    )
    parser.add_argument(
        '--feature', type=int, required=True,
        help='Feature ID to look up (signed integer, e.g., 199 or -199)'
    )
    parser.add_argument(
        '--dataset', default=None,
        help='Filter to specific dataset (optional)'
    )
    parser.add_argument(
        '--meta_dir', default='../data/processed_csvs_anon',
        help='Path to meta CSV directory (default: ../data/processed_csvs_anon)'
    )
    args = parser.parse_args()

    # Load qualitative analysis CSV
    df = pd.read_csv(args.csv)

    # Filter by dataset if specified
    if args.dataset:
        df = df[df['dataset'] == args.dataset]
        if df.empty:
            print(f"No rows found for dataset '{args.dataset}'")
            return

    # Parse feature_indices and find rows containing the target feature
    matching_rows = []
    for idx, row in df.iterrows():
        features = parse_feature_indices(row['feature_indices'])
        if args.feature in features:
            matching_rows.append(row)

    if not matching_rows:
        sign = '+' if args.feature >= 0 else '-'
        print(f"Feature {abs(args.feature)} ({sign}) not found in any voxels")
        return

    # Convert to DataFrame
    matches_df = pd.DataFrame(matching_rows)

    # Load metadata for each participant/dataset combination and join
    meta_dir = Path(args.meta_dir)
    results = []

    for _, row in matches_df.iterrows():
        participant = row['participant']
        dataset = row['dataset']
        neuroid = row['neuroid']
        nc_r = row['NC Normalized R']

        # Load corresponding meta CSV
        meta_path = meta_dir / participant / f"{dataset}_meta.csv"
        if not meta_path.exists():
            print(f"Warning: Meta file not found: {meta_path}")
            results.append({
                'Participant': participant,
                'Neuroid': neuroid,
                'Dataset': dataset,
                'NC Norm R': nc_r,
                'parc_lang': 'N/A',
                'parc_glasser': 'N/A',
                'parc_name_glasser': 'N/A'
            })
            continue

        meta_df = pd.read_csv(meta_path)

        # Find matching neuroid
        meta_row = meta_df[meta_df['neuroid_id'] == neuroid]

        if meta_row.empty:
            print(f"Warning: Neuroid {neuroid} not found in {meta_path}")
            results.append({
                'Participant': participant,
                'Neuroid': neuroid,
                'Dataset': dataset,
                'NC Norm R': nc_r,
                'parc_lang': 'N/A',
                'parc_glasser': 'N/A',
                'parc_name_glasser': 'N/A'
            })
            continue

        meta_row = meta_row.iloc[0]
        results.append({
            'Participant': participant,
            'Neuroid': neuroid,
            'Dataset': dataset,
            'NC Norm R': nc_r,
            'parc_lang': meta_row.get('parc_lang', 'N/A'),
            'parc_glasser': meta_row.get('parc_glasser', 'N/A'),
            'parc_name_glasser': meta_row.get('parc_name_glasser', 'N/A')
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print header
    sign = '+' if args.feature >= 0 else '-'
    print(f"\nFeature {abs(args.feature)} ({sign}) found in {len(results_df)} voxels:\n")

    # Print table
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
