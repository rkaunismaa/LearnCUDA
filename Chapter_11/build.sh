#!/bin/bash
# Chapter 11 setup script
# Installs Python packages and builds the custom CUDA extension.

set -e
VENV=/home/rob/PythonEnvironments/LearnCUDA/.learncuda

echo "Activating virtual environment: $VENV"
source "$VENV/bin/activate"

echo ""
echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing CuPy (CUDA 12.x)..."
pip install cupy-cuda12x

echo ""
echo "Installing helper packages..."
pip install numpy matplotlib jupyter

echo ""
echo "Building custom CUDA extension..."
cd 03_torch_custom_extension
pip install -e .
cd ..

echo ""
echo "Done! Run the examples:"
echo "  python 01_torch_cuda_basics.py"
echo "  python 02_cupy_basics.py"
echo "  python 03_torch_custom_extension/test_extension.py"
