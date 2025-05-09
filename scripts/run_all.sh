#!/bin/bash
# Run all validation experiments for H²-compressibility conjecture

# Set the device to use (cuda or cpu)
DEVICE="cuda"
if ! command -v nvidia-smi &> /dev/null; then
    DEVICE="cpu"
    echo "CUDA not available, using CPU"
fi

echo "Starting H²-compressibility validation..."
echo "Device: $DEVICE"
echo "------------------------------------"

# Create result directories if they don't exist
mkdir -p ../results
mkdir -p ../visualizations

# 1. Analyze rank of far-field blocks with different noise levels
echo "Running rank analysis..."
python analyze_rank.py --n 1024 --d 64 --block_size 64 --noise_levels 0.0 0.01 0.05 0.1 --device $DEVICE
echo "Rank analysis complete!"
echo "------------------------------------"

# 2. Test H² attention with different rank parameters
echo "Running H² attention tests..."
python h2_attention.py --n 1024 --d 64 --ranks 5 10 15 20 --noise_level 0.01 --device $DEVICE
echo "H² attention tests complete!"
echo "------------------------------------"

# 3. Test scaling properties with sequence length
echo "Running scaling tests..."
python scaling_test.py --d 64 --seq_lengths 512 1024 2048 4096 --rank 10 --device $DEVICE
echo "Scaling tests complete!"
echo "------------------------------------"

# 4. Analyze pre-trained model (optional - requires transformers)
if pip show transformers &> /dev/null; then
    echo "Running pre-trained model analysis..."
    python analyze_pretrained.py --model_name bert-base-uncased --max_length 256 --block_size 32 --num_samples 5 --device $DEVICE
    echo "Pre-trained model analysis complete!"
else
    echo "Transformers package not found, skipping pre-trained model analysis"
fi
echo "------------------------------------"

echo "All validation experiments complete!"
echo "Results and visualizations are available in the respective directories."
echo "Check the README.md file for more information."
