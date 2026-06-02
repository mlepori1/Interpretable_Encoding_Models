# Little script to combine results into one file

import os
import pandas as pd
import glob

# Get all per-run CSV files (exclude results.csv if it already exists)
csv_files = [f for f in glob.glob("*.csv") if f != "results.csv"]

# Read and concatenate
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# ---------------------------------------------------------------------------
# Deduplicate: remove voxels whose beta pattern is identical to another voxel
# in the same participant. This matches the df_filt_dedup logic in the analysis
# notebooks. We apply it to both base and gen neuroids.
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
# Script is 10 levels deep inside Neuroscope/results/...
project_root = os.path.abspath(os.path.join(script_dir, *[".."] * 10))
betas_base = os.path.join(project_root, "data", "processed_csvs_anon")


def get_unique_neuroids(participant, dataset):
    """Return the set of neuroid IDs with unique beta vectors for a participant."""
    betas_path = os.path.join(betas_base, participant, f"{dataset}_betas.csv")
    betas = pd.read_csv(betas_path, index_col=0)
    seen = {}
    unique_nids = set()
    for col in betas.columns:
        key = tuple(betas[col].values)
        if key not in seen:
            seen[key] = col
            unique_nids.add(int(col))
    return unique_nids


# Build unique-neuroid sets for all participants that appear in the data
unique_neuroids = {}
for (participant, dataset) in (
    df[["base_participant", "base_dataset"]].drop_duplicates().itertuples(index=False)
):
    if (participant, dataset) not in unique_neuroids:
        unique_neuroids[(participant, dataset)] = get_unique_neuroids(participant, dataset)
for (participant, dataset) in (
    df[["gen_participant", "gen_dataset"]].drop_duplicates().itertuples(index=False)
):
    if (participant, dataset) not in unique_neuroids:
        unique_neuroids[(participant, dataset)] = get_unique_neuroids(participant, dataset)

# Filter: keep rows where both base and gen neuroids are unique for their participant
base_keep = df.apply(
    lambda row: int(row["base_neuroid"]) in unique_neuroids[(row["base_participant"], row["base_dataset"])],
    axis=1,
)
gen_keep = df.apply(
    lambda row: int(row["gen_neuroid"]) in unique_neuroids[(row["gen_participant"], row["gen_dataset"])],
    axis=1,
)
df = df[base_keep & gen_keep].reset_index(drop=True)

# Save combined file
df.to_csv("results.csv", index=False)
