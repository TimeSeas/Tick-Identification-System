# Tick Detection Environment Setup Script
# Compatible with CUDA 12.4 and RTX 4060

$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Create virtual environment ===" -ForegroundColor Green
python -m venv tick_env
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

Write-Host "=== Step 2: Install PyTorch with CUDA 12.1 support ===" -ForegroundColor Green
.\tick_env\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "=== Step 3: Install Detectron2 ===" -ForegroundColor Green
pip install 'git+https://github.com/facebookresearch/detectron2.git'

Write-Host "=== Step 4: Install other dependencies ===" -ForegroundColor Green
pip install pycocotools opencv-python Pillow numpy

Write-Host "=== Step 5: Verify installation ===" -ForegroundColor Green
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"

Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host "To activate environment run: .\tick_env\Scripts\Activate.ps1"