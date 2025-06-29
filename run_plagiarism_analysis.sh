#!/bin/bash

# Simple script to run plagiarism detection on articles
# Usage: ./run_plagiarism_analysis.sh

# Default values - modify these according to your setup
MODEL_PATH="best_siamese_bert.pth"
ARTICLES_DIR="/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/2024_25/arxiv_data/full_papers"
EXTRACT_SCRIPT="extract_paragraphs.py"
OUTPUT_FILE="plagiarism_results.csv"
MAX_ARTICLES=2
SIMILARITY_THRESHOLD=0.7
MODE="cross"  # Options: cross, internal, both

# Help function
show_help() {
    echo "Plagiarism Detection Analysis"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model_path PATH         Path to trained model (default: $MODEL_PATH)"
    echo "  -a, --articles_dir DIR        Directory with .tex articles (default: $ARTICLES_DIR)"
    echo "  -e, --extract_script PATH     Path to extract_paragraphs.py (default: $EXTRACT_SCRIPT)"
    echo "  -o, --output_file FILE        Output file (default: $OUTPUT_FILE)"
    echo "  -n, --max_articles N          Max articles to analyze (default: $MAX_ARTICLES)"
    echo "  -t, --threshold FLOAT         Similarity threshold (default: $SIMILARITY_THRESHOLD)"
    echo "  -M, --mode MODE               Analysis mode: cross/internal/both (default: $MODE)"
    echo "  -h, --help                   Show this help"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with 2 articles"
    echo "  $0"
    echo ""
    echo "  # Analyze 5 articles with lower threshold"
    echo "  $0 --max_articles 5 --threshold 0.5"
    echo ""
    echo "  # Internal plagiarism analysis only"
    echo "  $0 --mode internal"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -a|--articles_dir)
            ARTICLES_DIR="$2"
            shift 2
            ;;
        -e|--extract_script)
            EXTRACT_SCRIPT="$2"
            shift 2
            ;;
        -o|--output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -n|--max_articles)
            MAX_ARTICLES="$2"
            shift 2
            ;;
        -t|--threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        -M|--mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if required files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found"
    exit 1
fi

if [ ! -d "$ARTICLES_DIR" ]; then
    echo "Error: Articles directory $ARTICLES_DIR not found"
    exit 1
fi

if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "Error: Extract script $EXTRACT_SCRIPT not found"
    exit 1
fi

if [ ! -f "apply_plagiarism_detection.py" ]; then
    echo "Error: apply_plagiarism_detection.py not found in current directory"
    exit 1
fi

echo "==================================="
echo "Plagiarism Detection Analysis"
echo "==================================="
echo "Model: $MODEL_PATH"
echo "Articles Directory: $ARTICLES_DIR"
echo "Max Articles: $MAX_ARTICLES"
echo "Similarity Threshold: $SIMILARITY_THRESHOLD"
echo "Analysis Mode: $MODE"
echo "Output File: $OUTPUT_FILE"
echo "==================================="
echo ""

# Run the analysis
python3 apply_plagiarism_detection.py \
    --model_path "$MODEL_PATH" \
    --articles_dir "$ARTICLES_DIR" \
    --extract_script "$EXTRACT_SCRIPT" \
    --similarity_threshold "$SIMILARITY_THRESHOLD" \
    --max_articles "$MAX_ARTICLES" \
    --output_file "$OUTPUT_FILE" \
    --mode "$MODE"

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Analysis completed successfully!"
    echo "==================================="
    echo "Results saved to:"
    echo "  - CSV: $OUTPUT_FILE"
    echo "  - JSON: ${OUTPUT_FILE%.csv}.json"
    echo ""
    
    # Show quick summary if CSV exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Quick summary:"
        echo "Total potential plagiarism instances: $(tail -n +2 "$OUTPUT_FILE" | wc -l)"
        echo ""
        echo "Top 5 highest similarity scores:"
        tail -n +2 "$OUTPUT_FILE" | sort -t',' -k5 -nr | head -5 | cut -d',' -f1,2,5 | while IFS=',' read -r art1 art2 score; do
            echo "  $art1 ↔ $art2: $score"
        done
    fi
else
    echo ""
    echo "Error: Analysis failed!"
    exit 1
fi
