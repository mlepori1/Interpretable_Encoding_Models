#!/bin/bash
# Run all 4 featurizer configs for langfroi12345(all) voxels
# Usage: bash run_langfroi12345.sh p3 p6
# Split 8 participants across 4 nodes (2 per node):
#   Node 1: bash run_langfroi12345.sh p3 p6
#   Node 2: bash run_langfroi12345.sh p5 p1
#   Node 3: bash run_langfroi12345.sh p7 p2
#   Node 4: bash run_langfroi12345.sh p4 p8

for participant in "$@"; do
    echo "=== Processing $participant ==="

    # 1. Log prob only
    echo "  Running logprobs_only..."
    python run_experiment_1.py \
        --datasets langfroi12345all \
        -p "$participant" \
        -n 10000 \
        --logprobs_only \
        --standardize_betas \
        --output_suffix langfroi12345all_20260311

    # Log prob with shuffle voxels
#    echo "  Running logprobs_only with shuffled voxels..."
#    python run_experiment_1.py \
#        --datasets langfroi12345all \
#        -p "$participant" \
#        -n 10000 \
#        --logprobs_only \
#        --standardize_betas \
#        --voxel_shuffle_control \
#        --output_suffix langfroi12345all_20260311

    # 2. Residual stream + logprobs
    echo "  Running hidden_states + logprobs..."
    python run_experiment_1.py \
        --datasets langfroi12345all \
        -p "$participant" \
        -n 10000 \
        --use_logprobs \
        --standardize_betas \
        --output_suffix langfroi12345all_20260311

    # Residual stream + logprobs with shuffle voxels
    echo "  Running hidden_states + logprobs with shuffled voxels..."
    python run_experiment_1.py \
        --datasets langfroi12345all \
        -p "$participant" \
        -n 10000 \
        --use_logprobs \
        --standardize_betas \
        --voxel_shuffle_control \
        --output_suffix langfroi12345all_20260311

    # 3. JumpReLU SAE + logprobs
    echo "  Running JumpReLU SAE..."
    python run_experiment_1.py \
        --datasets langfroi12345all \
        -p "$participant" \
        -n 10000 \
        --use_sae \
        --sae_release gemma-scope-2b-pt-res-canonical \
        --use_logprobs \
        --standardize_betas \
        --output_suffix langfroi12345all_20260311

    # 4. Matryoshka SAE + logprobs
    echo "  Running Matryoshka SAE..."
    python run_experiment_1.py \
        --datasets langfroi12345all \
        -p "$participant" \
        -n 10000 \
        --use_sae \
        --sae_release gemma-2-2b-res-matryoshka-dc \
        --use_logprobs \
        --standardize_betas \
        --output_suffix langfroi12345all_20260311

    # 5. Topic model without logprobs (content_only)
    echo "  Running topic model (no logprobs)..."
    python run_experiment_1.py \
        --datasets langfroi12345all \
        -p "$participant" \
        -n 10000 \
        --use_topic_model \
        --standardize_betas \
        --output_suffix langfroi12345all_20260311


    echo "=== Done with $participant ==="
done
