#!/bin/bash
echo "=== Backing up training results before exit ==="

# Copy all training results back to home
if [ -d "/tmp/training_work" ]; then
    echo "Copying training results from /tmp/training_work/ to home..."
    
    # Copy model files
    cp /tmp/training_work/*.pth ~/ 2>/dev/null
    cp /tmp/training_work/*.json ~/ 2>/dev/null
    cp /tmp/training_work/*.png ~/ 2>/dev/null
    cp /tmp/training_work/*.log ~/ 2>/dev/null
    cp /tmp/training_work/*.txt ~/ 2>/dev/null
    
    # Copy to plagiarism_output directory if it exists
    if [ -d "~/plagiarism_project/suspicious-document/plagiarism_output" ]; then
        echo "Also copying to plagiarism_output directory..."
        cp /tmp/training_work/*.pth ~/plagiarism_project/suspicious-document/plagiarism_output/ 2>/dev/null
        cp /tmp/training_work/*.png ~/plagiarism_project/suspicious-document/plagiarism_output/ 2>/dev/null
        cp /tmp/training_work/*.json ~/plagiarism_project/suspicious-document/plagiarism_output/ 2>/dev/null
    fi
    
    echo "Files copied to home directory:"
    ls -la ~/*.pth ~/*.png ~/*.json ~/training_*.log 2>/dev/null
    
    echo "Training artifacts backed up successfully!"
else
    echo "No training work directory found in /tmp/"
fi

echo "=== Training backup complete! Safe to exit. ==="
echo "Next training session: run ~/start_training_session.sh"

# Exit the GPU session
exit
