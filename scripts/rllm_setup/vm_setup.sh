#!/bin/bash
set -e

echo "Starting setup script..."

# Create conda environment with Python 3.12
echo "Creating conda environment 'cai' with Python 3.12..."
conda create -n cai python=3.12 -y

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cai

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Clone and install swe-smith
echo "Cloning and installing swe-smith..."
git clone https://github.com/threewisemonkeys-as/SWE-smith.git
cd SWE-smith
git checkout cai
pip install -e .
python swesmith/build_repo/download_images.py
cd ..

# Clone and install swe-agent
echo "Cloning and installing swe-agent..."
git clone https://github.com/threewisemonkeys-as/SWE-agent.git
cd SWE-agent
git checkout cai
pip install -e .
cd ..

# Clone and install swe-bench
echo "Cloning and installing swe-bench..."
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
pip install -e .
cd ..



