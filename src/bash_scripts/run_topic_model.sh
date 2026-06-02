export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Neuroscope/

layers=(12)
use_sae=("")
use_topic_model=("--use_topic_model")
standardize=("--standardize_features")
sae_release=("")
shuffle_control=("")

for i in "${!layers[@]}"; do
    export LAYER=${layers[i]}
    export USE_SAE=${use_sae[i]}
    export USE_TOPIC_MODEL=${use_topic_model[i]}
    export STANDARDIZE=${standardize[i]}
    export RELEASE=${sae_release[i]}
    export SHUFFLE_CONTROL=${shuffle_control[i]}

    echo "Running $LAYER $USE_SAE $STANDARDIZE $RELEASE $SHUFFLE_CONTROL $USE_TOPIC_MODEL"

    sbatch -o out/1_${i}.out -e err/1_${i}.err $PROJECT_DIR/src/run_exp1.script
done

