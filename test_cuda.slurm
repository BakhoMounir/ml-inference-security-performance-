#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:05:00
#SBATCH --output=test_cuda_%j.out
#SBATCH --error=test_cuda_%j.err

source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu_2.0.1

echo "HOST: $(hostname)"
echo "PYTHON: $(which python)"

nvidia-smi

python -c "import torch; print('torch:', torch.__version__); print('torch path:', torch.__file__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
