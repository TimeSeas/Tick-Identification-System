# Tick Detection Environment Setup
# Run this script to set up the environment

# Create virtual environment (in project directory)
python -m venv tick_env

# Activate (Windows PowerShell)
# .\tick_env\Scripts\Activate

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2 from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
pip install pycocotools opencv-python Pillow numpy tensorrt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"