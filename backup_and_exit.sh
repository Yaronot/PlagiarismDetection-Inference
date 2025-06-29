#!/bin/bash
echo "=== Backing up work before exit ==="

# Copy all results back to home
if [ -d "/tmp/plagiarism_work" ]; then
    echo "Copying results from /tmp/plagiarism_work/ to home..."
    cp /tmp/plagiarism_work/plagiarism_results.* ~/ 2>/dev/null
    cp /tmp/plagiarism_work/*.txt ~/ 2>/dev/null
    cp /tmp/plagiarism_work/*.log ~/ 2>/dev/null
    cp /tmp/plagiarism_work/*.out ~/ 2>/dev/null
    
    echo "Files copied to home directory:"
    ls -la ~/plagiarism_results.* ~/analysis_*.txt ~/verbose_*.txt 2>/dev/null
else
    echo "No work directory found in /tmp/"
fi

echo "=== Backup complete! Safe to exit. ==="
echo "Next session: run ~/start_gpu_session.sh"

# Exit the GPU session
exit
