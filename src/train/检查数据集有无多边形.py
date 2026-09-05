from pycocotools.coco import COCO
import numpy as np

# 替换成你的标注文件路径
ann_file = "tick-bite-improved.v7-random_split_resize_640.coco-segmentation/train/_annotations.coco.json"

# 加载 COCO 数据
coco = COCO(ann_file)

# 获取第一张图片的所有标注 ID
img_ids = list(coco.imgs.keys())
ann_ids = coco.getAnnIds(imgIds=img_ids[0])
anns = coco.loadAnns(ann_ids)

# 打印第一条标注的详细信息
first_ann = anns[0]
print("标注的 keys:", first_ann.keys())
print("segmentation 字段内容:", first_ann['segmentation'])
print("segmentation 的数据类型:", type(first_ann['segmentation']))