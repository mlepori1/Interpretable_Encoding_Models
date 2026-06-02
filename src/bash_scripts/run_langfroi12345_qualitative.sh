#!/bin/bash
# Run qualitative analysis for langfroi12345 voxels (Matryoshka SAE)
# Split 8 participants across 2 nodes (4 per node):
#   Node 1: bash run_langfroi12345_qualitative.sh p3 p6 p5 p1
#   Node 2: bash run_langfroi12345_qualitative.sh p4 p2 p7 p8

for participant in "$@"; do
    echo "=== Qualitative analysis for $participant ==="
    python run_qualitative_experiment.py \
        --datasets langfroi12345 \
        -p "$participant" \
        --per_participant \
        --all_voxels \
        --save_meta \
        --use_logprobs \
        --standardize_betas \
        --sae_release gemma-2-2b-res-matryoshka-dc \
        --results_suffix langfroi12345_20260311
    echo "=== Done with $participant ==="
done
