"""
多类别蜱种分类训练（通用版，类数由划分 JSON 决定，31 类用）
与 train_tick_species.py 的区别：
- 类数不再写死，由 split JSON 自动读取
- 长尾处理：WEIGHTED 按类频率倒数加权采样；LABEL_SMOOTHING 标签平滑
- INPUT_SIZE 可切 448（细粒度分类更高分辨率通常更准，但显存占用翻倍）
改顶部配置区的常量即可切换消融实验，无需命令行参数。
"""
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score,confusion_matrix
from torchvision.models import ResNet50_Weights

# ==================== 配置区 ====================
SPLIT_JSON="data/tick_species/split_clean.json"  # 划分文件，由 scripts/split_species.py 生成（清洗后 28 类）
OUTPUT_DIR="output/tick_species_clean_224"       # 输出目录（换实验时改名，避免覆盖）
INPUT_SIZE=224        # 输入分辨率：224 / 448（细粒度 448 更准，8GB 显存 batch 降到 16）
WEIGHTED=True         # 是否按类频率倒数加权采样（对抗长尾分布）
LABEL_SMOOTHING=0.1   # 标签平滑系数，0 为关闭（防止过拟合/过度自信）
BATCH_SIZE,LR,EPOCHS,SEED,NUM_WORKERS=32,1e-4,20,42,2
#批次大小（每次梯度更新所使用的样本数量）、学习率（控制参数更新的步长大小）、
#训练轮数（整个训练集被完整遍历的次数）、随机种子（保证划分/增强可复现）、用于加载的主进程数
# ================================================

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

class SplitDataset(torch.utils.data.Dataset):
    """自定义数据集类：读取划分 JSON 里的图片路径与类别索引，返回预处理后的图像张量"""
    def __init__(self,split_name,class_index,transform=None):
        # 从划分文件取出该 split 的条目，label 用 class_index 映射成数字（模型只认数字标签）
        data=json.load(open(SPLIT_JSON,encoding="utf-8"))[split_name]
        self.items=[(Path(d["path"]),class_index[d["label"]]) for d in data]
        self.transform=transform
    def __len__(self):return len(self.items)
    def __getitem__(self,index):
        # DataLoader 迭代时自动调用，组成 batch
        for _ in range(len(self)):#坏图跳过：依次尝试后续图片，最多找一圈，避免无限循环
            path,label=self.items[index]
            try:
                img=Image.open(path).convert("RGB")#读取图像，统一转 RGB（防止灰度/带 alpha 的图）
                return self.transform(img),label#返回增强后的张量与类别索引
            except Exception:
                index=(index+1)%len(self)
        raise RuntimeError("没有可打开的图片")

def get_model(num_classes):
    """加载 ImageNet 预训练 ResNet-50，替换最后的全连接层为当前类别数"""
    model=torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(model.fc.in_features,num_classes)#复用卷积层的通用图像特征，只微调分类头
    return model.to(device)

def train_one_epoch(model,loader,criterion,optimizer):
    """遍历一次训练集：前向传播→计算损失→反向传播→更新权重，返回平均损失与训练准确率"""
    model.train()
    total_loss,correct,n=0,0,0
    for imgs,labels in loader:
        imgs,labels=imgs.to(device),labels.to(device)
        optimizer.zero_grad()#清空上一批次的梯度，防止累积
        logits=model(imgs)#前向传播，logits 是每类原始分数
        loss=criterion(logits,labels)#交叉熵损失（可含标签平滑）
        loss.backward(); optimizer.step()#反向传播计算梯度 + 按梯度更新权重
        total_loss+=loss.item()*len(imgs)#按样本数累加，最后算加权平均
        correct+=(logits.argmax(1)==labels).sum().item()#argmax 取分数最高的类，与真实标签比对
        n+=len(imgs)
    return total_loss/n,correct/n

@torch.no_grad()#关闭梯度计算，节省显存并加速
def evaluate(model,loader):
    """评估一个数据集：返回准确率、macro-F1、预测列表、真实标签列表、每样本最高概率"""
    model.eval()#切换到评估模式（影响 BatchNorm/Dropout 行为）
    all_preds,all_labels,all_probs=[],[],[]
    correct,n=0,0
    for imgs,labels in loader:
        imgs,labels=imgs.to(device),labels.to(device)
        logits=model(imgs)
        preds=logits.argmax(dim=1)#取概率最大的类作为预测
        correct+=(preds==labels).sum().item(); n+=len(imgs)
        all_preds+=preds.cpu().tolist()#.cpu() 把 GPU 张量搬到内存再转列表
        all_labels+=labels.cpu().tolist()
        all_probs.append(logits.softmax(dim=1).cpu())#softmax 转概率（阈值表要用到置信度）
    probs=torch.cat(all_probs).max(dim=1).values.tolist()#每样本取它预测类的概率
    #macro-F1 对每类分别算再平均，类别不平衡时比 acc 更能反映小类表现
    return correct/n,f1_score(all_labels,all_preds,average='macro'),all_preds,all_labels,probs

def save_val_csv(dataset,preds,labels,probs,classes):
    """保存验证集预测 CSV（fname,true,pred,prob）——scripts/confidence_thresholds.py 的输入"""
    with open(f"{OUTPUT_DIR}/val_predictions.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["fname","true","pred","prob"])
        for (path,_),t,p,prob in zip(dataset.items,labels,preds,probs):
            w.writerow([Path(path).name,classes[t],classes[p],round(prob*100,2)])

def plot_confusion_matrix(preds,labels,classes):
    """绘制并保存混淆矩阵：行=真实，列=预测，对角线是分对的；某行全暗说明该类全错"""
    plt.rcParams['font.sans-serif']=['SimHei']#中文字体，否则"预测/真实"显示为方块
    plt.rcParams['axes.unicode_minus']=False
    cm=confusion_matrix(labels,preds)
    plt.figure(figsize=(max(8,len(classes)*0.55),max(8,len(classes)*0.5)))#图大小随类数自适应
    plt.imshow(cm,cmap="Blues")#颜色越深=数量越多
    if len(classes)<=10:#类少时逐格标数字；类多（31 类）标了也看不清，只靠颜色深浅
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j,i,cm[i,j],ha="center",va="center",fontsize=8)
    plt.xticks(range(len(classes)),classes,rotation=45,ha="right",fontsize=7)
    plt.yticks(range(len(classes)),classes,fontsize=7)
    plt.xlabel("预测"); plt.ylabel("真实"); plt.colorbar()
    plt.tight_layout()#自动压缩边距，防止长标签被裁掉
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png",dpi=150,bbox_inches="tight")

def main():
    print(device,"| weighted:",WEIGHTED,"| label_smoothing:",LABEL_SMOOTHING,"| size:",INPUT_SIZE)
    os.makedirs(f"{OUTPUT_DIR}/model",exist_ok=True)#输出目录先建好，避免保存时报错
    model_path=f"{OUTPUT_DIR}/model/tick_species_best.pth"

    #从划分文件读出全部类别（按字母序），建立"类别名→数字索引"映射
    split=json.load(open(SPLIT_JSON,encoding="utf-8"))
    classes=sorted({d["label"] for d in split["train"]})
    class_index={c:i for i,c in enumerate(classes)}
    print(f"{len(classes)} 类")

    #图像预处理：ImageNet 标准归一化（预训练权重是按这个统计量训练的，必须一致）
    normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    train_tf=T.Compose([
        T.RandomResizedCrop(INPUT_SIZE,scale=(0.7,1.0),ratio=(0.75,1.3333)),
        #随机裁一块（原图 70%-100% 面积）再缩放。蜱虫主体占图大，下限 0.7 防止把虫裁掉
        T.RandomHorizontalFlip(p=0.5),#左右镜像。蜱虫左右对称，翻转不影响类别
        T.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        #亮度/对比度/饱和度各 ±20% 扰动，模拟不同光照（iNaturalist 照片光照差异大）
        T.ToTensor(),normalize])
    #评估不用随机增强：先等比放大到 1.14 倍再中心裁剪，与训练时的构图保持一致
    eval_tf=T.Compose([T.Resize(int(INPUT_SIZE*1.14)),T.CenterCrop(INPUT_SIZE),T.ToTensor(),normalize])

    train_ds=SplitDataset("train",class_index,train_tf)
    val_ds=SplitDataset("val",class_index,eval_tf)
    test_ds=SplitDataset("test",class_index,eval_tf)

    loader_kwargs=dict(batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    if WEIGHTED:#长尾处理：权重=1/类样本数，小类被采到的机会与大类相当
        counts=np.bincount([i for _,i in train_ds.items])
        weights=torch.tensor([1.0/counts[i] for _,i in train_ds.items],dtype=torch.float)
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(weights))
        train_loader=torch.utils.data.DataLoader(train_ds,sampler=sampler,**loader_kwargs)
        #用了 sampler 就不能再开 shuffle（两者互斥）
    else:
        train_loader=torch.utils.data.DataLoader(train_ds,shuffle=True,**loader_kwargs)
    val_loader=torch.utils.data.DataLoader(val_ds,shuffle=False,**loader_kwargs)
    test_loader=torch.utils.data.DataLoader(test_ds,shuffle=False,**loader_kwargs)
    #验证/测试不打乱：保证 evaluate 里的预测与 dataset.items 顺序一致（存 CSV 要用路径对齐）

    model=get_model(len(classes))
    criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    #余弦退火：学习率按余弦曲线从初始值平滑降到 0，比固定学习率更容易收敛到好的极小值

    best_f1=0
    for epoch in range(EPOCHS):
        loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer)
        val_acc,val_f1,_,_,_=evaluate(model,val_loader)
        scheduler.step()#每个 epoch 结束后更新学习率
        print(f"epoch {epoch+1}/{EPOCHS} | loss {loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")
        if (epoch+1)%10==0:
            torch.save(model.state_dict(),f"{OUTPUT_DIR}/model/epoch_{epoch+1}.pth")
        if val_f1>best_f1:#用验证集 F1 选最优模型（类别不平衡时比 acc 合理）
            best_f1=val_f1
            torch.save(model.state_dict(),model_path)

    #用最优模型做最终评估
    model.load_state_dict(torch.load(model_path,weights_only=True))
    val_acc,val_f1,v_p,v_l,v_prob=evaluate(model,val_loader)
    save_val_csv(val_ds,v_p,v_l,v_prob,classes)#验证集预测 CSV，供阈值复现脚本使用
    test_acc,test_f1,t_p,t_l,_=evaluate(model,test_loader)
    plot_confusion_matrix(t_p,t_l,classes)
    print(f"best val_f1: {best_f1:.4f} | test_acc: {test_acc:.4f} | test_f1: {test_f1:.4f}")

if __name__=="__main__":
    main()
