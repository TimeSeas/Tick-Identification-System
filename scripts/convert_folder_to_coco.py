"""
Convert folder-structured insect images to COCO format for detection training.

Usage:
    python scripts/convert_to_coco.py --input archive2/dataset/train --output dataset_coco.json

This script assumes images are organized as:
    archive2/dataset/train/
        tick/
            img1.jpg
            img2.jpg
        mite/
            img1.jpg
            img2.jpg
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def create_coco_structure():
    """Create base COCO structure."""
    return {
        "info": {
            "description": "Insect Detection Dataset",
            "version": "1.0",
            "year": 2026
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

def add_category(categories, name):
    """Add a category if not exists, return id."""
    for cat in categories:
        if cat['name'] == name:
            return cat['id']

    cat_id = len(categories) + 1
    categories.append({
        "id": cat_id,
        "name": name,
        "supercategory": "insect"
    })
    return cat_id

def convert_folder_to_coco(input_dir, output_file, categories_list=None):
    """
    Convert folder-structured images to COCO format.

    Args:
        input_dir: Root directory with subfolders as class labels
        output_file: Output JSON file path
        categories_list: Optional list of specific categories to include
    """
    coco = create_coco_structure()

    # Define category mapping (folder name -> category name)
    category_mapping = {
        'tick': 'tick',
        'mite': 'mite',
        'ant': 'ant',
        'bee': 'bee',
        'beetle': 'beetle',
        'spider': 'spider'
    }

    ann_id = 1
    img_id = 1

    # Walk through input directory
    input_path = Path(input_dir)

    for class_folder in sorted(input_path.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name.lower()
        if class_name not in category_mapping:
            continue

        cat_name = category_mapping[class_name]
        cat_id = add_category(coco['categories'], cat_name)

        print(f"Processing {class_name}: {cat_name} (id={cat_id})")

        # Process each image
        for img_file in sorted(class_folder.iterdir()):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue

            try:
                # Get image dimensions
                with Image.open(img_file) as img:
                    width, height = img.size

                # Add image entry
                coco['images'].append({
                    "id": img_id,
                    "file_name": img_file.name,
                    "width": width,
                    "height": height,
                    "license": 0,
                    "category_id": cat_name
                })

                # Create full box annotation (entire image as bounding box)
                # This serves as a placeholder - manual annotation should refine this
                coco['annotations'].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [0, 0, width, height],  # Full image bbox
                    "area": width * height,
                    "iscrowd": 0,
                    "segmentation": [[0, 0, width, 0, width, height, 0, height]]
                })

                ann_id += 1
                img_id += 1

            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\nConverted {img_id - 1} images, {ann_id - 1} annotations")
    print(f"Categories: {[c['name'] for c in coco['categories']]}")
    print(f"Output: {output_file}")

    return coco

def main():
    parser = argparse.ArgumentParser(description="Convert folder dataset to COCO format")
    parser.add_argument("--input", "-i", required=True, help="Input directory with class subfolders")
    parser.add_argument("--output", "-o", default="dataset_coco.json", help="Output JSON file")
    parser.add_argument("--categories", "-c", nargs="+",
                        default=['tick', 'mite', 'ant', 'bee', 'beetle'],
                        help="Categories to include")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return

    convert_folder_to_coco(args.input, args.output)

if __name__ == "__main__":
    main()