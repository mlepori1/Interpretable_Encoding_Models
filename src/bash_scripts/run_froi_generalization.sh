export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Neuroscope/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SRC_DIR" || exit 1

use_sae=("--use_sae")
sae_release=("--sae_release gemma-2-2b-res-matryoshka-dc")
participants=("p3" "p6" "p5" "p1" "p4" "p2" "p7" "p8")
for i in "${!participants[@]}"; do
    for j in "${!participants[@]}"; do
        export USE_SAE=${use_sae[0]}
        export RELEASE=${sae_release[0]}
        export BASE_PARTICIPANT=${participants[i]}
        export GEN_PARTICIPANT=${participants[j]}

        echo "Running $USE_SAE $RELEASE $BASE_PARTICIPANT $GEN_PARTICIPANT"

        sbatch -o out/froi_generalization_${i}_${j}.out -e err/froi_generalization_${i}_${j}.err $PROJECT_DIR/src/bash_scripts/run_froi_generalization.script
    done
done

