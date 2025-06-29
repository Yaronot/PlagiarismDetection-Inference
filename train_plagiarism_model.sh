#!/bin/bash

# Training wrapper script with common configurations
echo "=== Plagiarism Model Training Wrapper ==="

# Parse arguments
SAMPLE_SIZE=2000
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
OUTPUT_DIR="/tmp/training_work"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sample_size N      Sample size for training (default: 2000)"
            echo "  --epochs N           Number of epochs (default: 5)"
            echo "  --batch_size N       Batch size (default: 16)"
            echo "  --learning_rate F    Learning rate (default: 2e-5)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Update the corpus path to point to our temporary location
CORPUS_PATH="/tmp/training_work/corpus"

# Run training
python3 plagiarism_detector.py \
    --corpus_path "$CORPUS_PATH" \
    --csv_file all_plagiarism_mappings.csv \
    --sample_size $SAMPLE_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Training Complete! ==="
echo "Results saved in: $OUTPUT_DIR"
echo "Don't forget to run: ~/backup_training_and_exit.sh"
