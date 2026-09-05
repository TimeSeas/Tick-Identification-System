"""Train Mask R-CNN for tick detection using MMDetection."""

from mmengine.config import Config
from mmengine.runner import Runner

# Config file
config_file = 'configs/mask_rcnn_ticks.py'

# Load config
cfg = Config.fromfile(config_file)

# Training
runner = Runner.from_cfg(cfg)
runner.train()