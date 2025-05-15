#!/bin/bash

# List of datasets
datasets=(
    "MNIST" "fashionMNIST" "DermaMNIST" "PneumoniaMNIST" "RetinaMNIST" 
    "BreastMNIST" "BloodMNIST" "OrganCMNIST" "OrganSMNIST" "german-credit"
)

# List of SENSE configuration scripts
sense_configs=(
    "Pointwise_Full.py"
    "Multisite_Full.py"
    "Multisite_Partial.py"
)

# Partition types
partitions=("iid" "balanced" "unbalanced")

# Output directory
RESULT_DIR="./results"

# Seed for reproducibility
SEED=42

for dataset in "${datasets[@]}"; do
    for partition in "${partitions[@]}"; do
        for script in "${sense_configs[@]}"; do

            # Extract config name from script name (remove 'sense_' prefix and '.py' suffix)
            config_name="${script//sense_/}"
            config_name="${config_name//.py/}"

            # Create directory for results
            OUTPUT_DIR="${RESULT_DIR}/${dataset}/${partition}/${config_name}"
            mkdir -p "$OUTPUT_DIR"

            # Build command
            cmd="python3 $script --dataset_name $dataset --seed $SEED"

            # Add partition type flags
            if [ "$partition" == "iid" ]; then
                cmd+=" --iid"
            elif [ "$partition" == "balanced" ]; then
                cmd+=" --balanced"
            elif [ "$partition" == "unbalanced" ]; then
                cmd+=" --unbalanced"
            fi

            # Output path
            OUTPUT_FILE="${OUTPUT_DIR}/output.txt"
            echo "🔁 Running: $dataset | $partition | $config_name"
            echo "$cmd" > "$OUTPUT_FILE"
            eval "$cmd" >> "$OUTPUT_FILE" 2>&1

        done
    done
done
