"""
主要训练蜱虫种类的分类，有3种蜱虫的数据集，训练完后应能输出蜱虫种类
用ResNet-50骨干网络
"""
import os
import torch.utils.data
import json
from pathlib import Path
from PIL import Image

output_model_p="output/tick_species_3cls.pth"
CLASSES=["dermacentor_variabilis", "ixodes_scapularis", "amblyomma_americanum"]
num_classes=3
split_json="data/tick_species/split_3cls.json"

BATCH_SIZE,LR,EPOCHS,SEED=32,1e-4,20,42

class SplitDataset(torch.utils.data.Dataset):
    #自定义数据集
    def __init__(self,split_name,transform=None):
        data=json.load(open(split_json,encoding="utf-8"))  [split_name]#导入json到data
        self.items=[(Path(d["path"]),CLASSES.index(d["label"])) for d in data]  #获取json中图片路径
        self.transform=transform
    def __len__(self):return len(self.items)
    def __getitem__(self, index):
        img=Image.open(self.items[index][0]).convert("RGB")
        return self.transform(img),self.items[index][1]

def main():
    """主函数"""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备：{device}")
    


if __name__=="__main__":
    main()