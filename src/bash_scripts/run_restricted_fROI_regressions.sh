export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Neuroscope/

min_features=("" "--min_feature_index 128" "--min_feature_index 128" "--min_feature_index 512" "--min_feature_index 2048" "--min_feature_index 8192")
max_features=("--max_feature_index 128" "" "--max_feature_index 512" "--max_feature_index 2048" "--max_feature_index 8192" "")

for i in "${!min_features[@]}"; do
    export MIN_FEATURE=${min_features[i]}
    export MAX_FEATURE=${max_features[i]}

    echo "Running $MIN_FEATURE $MAX_FEATURE"

    sbatch -o out/1_${i}.out -e err/1_${i}.err $PROJECT_DIR/src/bash_scripts/run_restricted_fROI_regressions.script
done

