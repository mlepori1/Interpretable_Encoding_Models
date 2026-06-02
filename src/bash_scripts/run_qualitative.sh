export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Neuroscope/

sae_release=("--sae_release gemma-scope-2b-pt-res-canonical" "--sae_release gemma-2-2b-res-matryoshka-dc")

for i in "${!sae_release[@]}"; do
    export RELEASE=${sae_release[i]}

    echo "Running  $RELEASE"

    sbatch -o out/1_${i}.out -e err/1_${i}.err $PROJECT_DIR/src/bash_scripts/run_qualitative.script
done

