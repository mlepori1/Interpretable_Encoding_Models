export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Neuroscope/

layers=(12 12 12 12 12 14 14 14 12)
use_sae=("" "" "" "--use_sae" "--use_sae" "" "--use_sae" "--use_sae" "--logprobs_only")
standardize=("" "" "--standardize_features" "" "" "" "" "" "")
sae_release=("" "" "" "--sae_release gemma-scope-2b-pt-res-canonical" "--sae_release gemma-2-2b-res-matryoshka-dc" "" "--sae_release gemma-scope-2b-pt-res-canonical" "--sae_release gemma-2-2b-res-matryoshka-dc" "")
shuffle_control=("--voxel_shuffle_control" "" "" "" "" "" "" "" "")

for i in "${!layers[@]}"; do
    export LAYER=${layers[i]}
    export USE_SAE=${use_sae[i]}
    export STANDARDIZE=${standardize[i]}
    export RELEASE=${sae_release[i]}
    export SHUFFLE_CONTROL=${shuffle_control[i]}

    echo "Running $LAYER $USE_SAE $STANDARDIZE $RELEASE $SHUFFLE_CONTROL"

    sbatch -o out/1_${i}.out -e err/1_${i}.err $PROJECT_DIR/src/bash_scripts/run_category_regressions.script
done

