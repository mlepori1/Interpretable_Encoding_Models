export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Neuroscope/

use_sae=("--use_sae")
sae_release=("--sae_release gemma-2-2b-res-matryoshka-dc")
results_fname=("--results_fname ghost.csv")

for i in "${!use_sae[@]}"; do
    export USE_SAE=${use_sae[i]}
    export RELEASE=${sae_release[i]}
    export RESULTS_FNAME=${results_fname[i]}
    echo "Running $USE_SAE $RELEASE $RESULTS_FNAME"

    sbatch -o out/2_${i}.out -e err/2_${i}.err $PROJECT_DIR/src/bash_scripts/run_ghost_generalization.script
done

