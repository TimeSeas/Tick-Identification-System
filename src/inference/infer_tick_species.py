"""
蜱种分类推理演示：对测试集图片做推理，生成预览图
每张预览图标注"真实的蜱种"与"模型推理的蜱种"（含中文名），分对绿色、分错红色
同时输出汇总 CSV（fname/真实/预测/置信度/是否正确）便于统计
改顶部常量切换模型（3 类 / 31 类）
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet50_Weights

# ==================== 配置区 ====================
MODEL_PATH="output/tick_species_31_weighted/model/tick_species_best.pth"  # 最优模型路径
SPLIT_JSON="data/tick_species/split_31cls.json"                            # 划分文件（测试集从这里取）
CN_TABLE="data/tick_species/物种中文名对照表.csv"                          # 学名→中文名对照表
OUT_DIR="output/tick_species_31_weighted/test_previews"                    # 预览图输出目录
INPUT_SIZE=224          # 与训练时一致
MAX_IMAGES=30           # 最多生成多少张预览（测试集全量太大，随机抽样）
SEED=7                  # 抽样种子，可复现
# ================================================

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取学名→中文名映射（标注/展示时用中文输出品种）
cn_map={}
with open(CN_TABLE,encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        cn_map[row["文件夹名(slug)"]]=row["中文名"]

def cn_of(slug):
    """slug → 中文名，缺省时回退学名"""
    return cn_map.get(slug,slug)

def load_font(size=20):
    """加载中文字体（Windows 自带的黑体），失败则回退 PIL 默认字体"""
    for fp in ["C:/Windows/Fonts/simhei.ttf","C:/Windows/Fonts/msyh.ttc"]:
        if Path(fp).exists():
            return ImageFont.truetype(fp,size)
    return ImageFont.load_default()

def main():
    import os
    os.makedirs(OUT_DIR,exist_ok=True)

    # 从划分文件读出类别，与训练脚本同一方式（字母序）
    split=json.load(open(SPLIT_JSON,encoding="utf-8"))
    classes=sorted({d["label"] for d in split["train"]})
    class_index={c:i for i,c in enumerate(classes)}
    test_items=[(Path(d["path"]),class_index[d["label"]]) for d in split["test"]]

    # 随机抽样（固定种子可复现）
    rng=random.Random(SEED)
    sampled=rng.sample(test_items,min(MAX_IMAGES,len(test_items)))

    # 评估变换：与训练脚本的 eval_tf 完全一致
    normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    eval_tf=T.Compose([T.Resize(int(INPUT_SIZE*1.14)),T.CenterCrop(INPUT_SIZE),
                       T.ToTensor(),normalize])

    # 加载最优模型
    model=torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(model.fc.in_features,len(classes))
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device,weights_only=True))
    model.to(device).eval()

    font_big=load_font(26)
    font_small=load_font(18)

    summary_path=f"{OUT_DIR}/summary.csv"
    rows=[]; correct_total=0
    with torch.no_grad():
        for path,true_idx in sampled:
            img=Image.open(path).convert("RGB")
            tensor=eval_tf(img).unsqueeze(0).to(device)
            probs=model(tensor).softmax(dim=1)[0]
            pred_idx=int(probs.argmax())
            prob=float(probs[pred_idx])
            is_correct=(pred_idx==true_idx)
            correct_total+=is_correct

            # 画预览图：放大 2 倍便于观察，顶部真实标签、底部预测标签
            canvas=img.resize((img.width*2,img.height*2))
            draw=ImageDraw.Draw(canvas)
            true_text=f"真实: {classes[true_idx]} ({cn_of(classes[true_idx])})"
            pred_text=f"预测: {classes[pred_idx]} ({cn_of(classes[pred_idx])}) {prob*100:.1f}%"
            # 分对：预测行绿色；分错：红色（真实行也标红提醒）
            pred_color=(34,139,34) if is_correct else (220,20,60)
            true_color=(0,0,0) if is_correct else (220,20,60)
            draw.rectangle([0,0,canvas.width,56],fill=(255,255,255,220))
            draw.rectangle([0,canvas.height-56,canvas.width,canvas.height],fill=(255,255,255,220))
            draw.text((12,14),true_text,fill=true_color,font=font_big)
            draw.text((12,canvas.height-44),pred_text,fill=pred_color,font=font_big)
            canvas.save(f"{OUT_DIR}/{path.stem}.jpg",quality=92)

            rows.append([path.name,classes[true_idx],cn_of(classes[true_idx]),
                         classes[pred_idx],cn_of(classes[pred_idx]),
                         round(prob*100,2),is_correct])

    with open(summary_path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["fname","真实slug","真实中文名","预测slug","预测中文名","置信度%","是否正确"])
        w.writerows(rows)

    n=len(sampled)
    print(f"抽样 {n} 张 | 正确 {correct_total} | 准确率 {correct_total/n*100:.1f}%")
    print(f"预览图 -> {OUT_DIR}/ | 汇总 CSV -> {summary_path}")

if __name__=="__main__":
    main()
