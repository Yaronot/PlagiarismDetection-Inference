#!/bin/bash
echo "=== Training Status Check ==="

# Check if training session is running
if pgrep -f "plagiarism_detector.py" > /dev/null; then
    echo "✓ Training process is running"
    echo "Process details:"
    ps aux | grep plagiarism_detector.py | grep -v grep
else
    echo "❌ No training process found"
fi

# Check GPU usage
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

# Check training files
echo ""
echo "Training Files:"
if [ -d "/tmp/training_work" ]; then
    echo "Work directory exists:"
    ls -la /tmp/training_work/*.pth /tmp/training_work/*.png /tmp/training_work/*.json 2>/dev/null || echo "No training outputs yet"
else
    echo "No training work directory found"
fi

echo ""
echo "Recent log entries:"
if [ -f "/tmp/training_work/training.log" ]; then
    tail -10 /tmp/training_work/training.log
else
    echo "No training log found"
fi
