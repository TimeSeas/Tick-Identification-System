# Tick Detection Environment Setup - Anaconda Version
# For RTX 4060 with CUDA 12.4

$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Create conda environment with Python 3.10 ===" -ForegroundColor Green
conda create -n tick_detection python=3.10 -y

Write-Host "=== Step 2: Activate environment ===" -ForegroundColor Green
conda activate tick_detection

Write-Host "=== Step 3: Install PyTorch with CUDA 12.1 support ===" -ForegroundColor Green
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "=== Step 4: Install Detectron2 ===" -ForegroundColor Green
pip install 'git+https://github.com/facebookresearch/detectron2.git'

Write-Host "=== Step 5: Install other dependencies ===" -ForegroundColor Green
pip install pycocotools opencv-python Pillow numpy

Write-Host "=== Step 6: Verify installation ===" -ForegroundColor Green
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host "To activate later run: conda activate tick_detection"