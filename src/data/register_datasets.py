"""Register tick detection datasets in Detectron2 format."""

from detectron2.data.datasets import register_coco_instances
import os

def register_tick_datasets():
    """Register tick detection datasets in Detectron2 format."""
    dataset_root = "ticks image detection.v5i.coco"

    # Register training set
    register_coco_instances(
        "tick_train",
        {},
        os.path.join(dataset_root, "train", "_annotations.coco.json"),
        os.path.join(dataset_root, "train")
    )

    # Register validation set
    register_coco_instances(
        "tick_val",
        {},
        os.path.join(dataset_root, "valid", "_annotations.coco.json"),
        os.path.join(dataset_root, "valid")
    )

    # Register test set
    register_coco_instances(
        "tick_test",
        {},
        os.path.join(dataset_root, "test", "_annotations.coco.json"),
        os.path.join(dataset_root, "test")
    )

if __name__ == "__main__":
    register_tick_datasets()
    print("Datasets registered successfully")
    from detectron2.data import DatasetCatalog
    print(f"Available datasets: {list(DatasetCatalog.keys())}")