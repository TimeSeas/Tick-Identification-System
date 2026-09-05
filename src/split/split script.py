"""
目标：把数据集最多的三个物种文件夹的图片按8:1:1分层划分为train/val/test，输出JSON清单
用sklearn.model_selection.train_test_split的stratify参数做分层抽样
"""
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import pandas as pd

ROOT=Path("data/tick_species")
df=pd.read_csv(f'{ROOT}/物种中文名对照表.csv')
result=df.sort_values("图片数",ascending=False)
top3_result=result.head(3)
SPECIES=top3_result['文件夹名(slug)'].tolist()

items=[]#每张图片对应一个类别
for sp in SPECIES:
    for img in sorted((ROOT/sp).glob("*.jpg")):#拼接路径后，获取所有jpg图片，并排序，添加到items列表
        items.append((img.as_posix(),sp))#.as_posix()以正斜杆/存路径,json跨平台可读
path=[p for p, _ in items]
labels=[l for _ ,l in items]

train_p,rest_p,train_l,rest_l=train_test_split(path,labels,test_size=0.2,stratify=labels,random_state=42)
#把所有图片列表切80%做训练集,并且设置分层抽样按类别比例，保证训练集和其余都是原来的类别比例。
#随机数种子保证能每次随机都能复现

test_p,val_p,test_l,val_l=train_test_split(rest_p,rest_l,test_size=0.5,stratify=rest_l,random_state=42)
#剩下20%的图片各分一半做测试集和验证集

spilt={
    "train":[{"path":p,"label":l} for p,l,in zip(train_p,train_l)],
    "val"  :[{"path":p,"label":l} for p,l,in zip(val_p,val_l)],
    "test" :[{"path":p,"label":l} for p,l,in zip(test_p,test_l)],
}
json.dump(spilt,open(f"{ROOT}/split_3cls.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)