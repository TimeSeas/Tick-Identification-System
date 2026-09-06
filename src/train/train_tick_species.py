"""
主要训练蜱虫种类的分类，有3种蜱虫的数据集，训练完后应能输出蜱虫种类
用ResNet-50骨干网络
"""
import os
import random
from matplotlib import pyplot as plt
import torch.utils.data
import json
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import f1_score,confusion_matrix
from PIL import Image
import numpy as np
import torchvision.transforms as T
output_dir="output/tick_species_3_results"
output_model_p=f"{output_dir}/model/tick_species_3cls.pth"
CLASSES=["dermacentor_variabilis", "ixodes_scapularis", "amblyomma_americanum"]
num_classes=3
split_json="data/tick_species/split_3cls.json"

BATCH_SIZE,LR,EPOCHS,SEED,NUM_WORKERS=32,1e-4,20,42,2
#批次大小（每次梯度更新所使用的样本数量）、学习率（控制参数更新的步长大小）、
#训练轮数（整个训练集被完整遍历的次数）、随机种子、用于加载的主进程数
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
class SplitDataset(torch.utils.data.Dataset):
    #自定义数据集类，返回原预处理过的图像张量
    def __init__(self,split_name,transforms=True):
        data=json.load(open(split_json,encoding="utf-8"))  [split_name]#提取是训练、测试、验证集的数据
        self.items=[(Path(d["path"]),CLASSES.index(d["label"])) for d in data]  #获取具体图片路径和类别
        self.transform=transforms
            
    def __len__(self):return len(self.items)
    def __getitem__(self, index):#DataLoader 在迭代时会自动调用 __getitem__来获取每个样本，然后组成 batch
        for _ in range(len(self)):#坏图跳过：最多尝试一圈，避免无限递归
            path,label_index=self.items[index]
            try:
                img=Image.open(path).convert("RGB")#读取实际图像
                return self.transform(img),label_index#返回经过预处理的图像张量，可输入到神经网络中进行计算
            except Exception:
                index=(index+1)%len(self)
        raise RuntimeError("整个数据集没有可打开的图片")


def get_model():
    model=torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)#加载预训练模型
    model.fc = nn.Linear(model.fc.in_features, num_classes)#替换神经网络的全连接层,复用模型卷积层的通用图像特征
    return model.to(device)
def train_one_epoch(model,loader,criterion,optimizer):
    """遍历一次数据集，做一次训练，计算损失，更新模型权重"""
    model.train()
    total_loss,correct,n=0,0,0
    for imgs,labels in loader:
        imgs,labels= imgs.to(device),labels.to(device)

        optimizer.zero_grad()#清空上一批次的梯度（防止累积）
        logits=model(imgs)
        loss=criterion(logits,labels)#用损失函数计算损失
        loss.backward()#计算总损失对模型所有可训练参数的梯度。
        optimizer.step()#根据梯度更新权重。

        total_loss+=loss.item()*len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(imgs)
    return total_loss / n, correct / n

@torch.no_grad() #关闭梯度计算，节省显存
def evaluate(model,loader):
    model.eval()#切换到评估模式
    all_preds,all_labels,all_probs=[],[],[]
    correct,n=0,0
    for imgs, labels in loader:
        imgs,labels =imgs.to(device),labels.to(device)
        logits=model(imgs)# 前向传播
        preds=logits.argmax(dim=1)#取概率最大的类作为预测
        correct+=(preds==labels).sum().item()
        n+=len(imgs)
        all_preds+=preds.cpu().tolist() #转为Python列表
        all_labels+=labels.cpu().tolist()#.cpu()从GPU搬到内存
        all_probs.append(logits.softmax(dim=1).cpu())#保留每个样本的类别概率（阈值表用）
    acc= correct/n
    f1=f1_score(all_labels,all_preds,average='macro')
    probs=torch.cat(all_probs).max(dim=1).values.cpu().tolist() if all_probs else []
    return acc, f1, all_preds, all_labels, probs

def plot_confusion_matrix(all_preds,all_labels,class_names,save_path):
    """绘制混淆矩阵
    对角线是分对的，非对角线是分错的
    如果某一行都很暗，就是某类全分错了，数据有问题
    """
    plt.rcParams['font.sans-serif']=['SimHei']#中文字体，否则"预测/真实"显示为方块
    plt.rcParams['axes.unicode_minus']=False
    cm = confusion_matrix(all_labels, all_preds)   # shape (3,3)，行=真实，列=预测
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")                   # 颜色越深 = 数量越多
    for i in range(len(cm)):                       # 在每个格子里标数字
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks(range(3), class_names, rotation=30)
    plt.yticks(range(3), class_names)
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.savefig(f"{save_path}/confusion_matrix", dpi=150, bbox_inches="tight")


def save_val_csv(dataset,preds,labels,probs,out_path):
    """保存验证集预测结果 CSV（fname,true,pred,prob），供阈值复现脚本使用"""
    import csv
    with open(out_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["fname","true","pred","prob"])
        for (path,_),t,p,prob in zip(dataset.items,labels,preds,probs):
            w.writerow([Path(path).name,CLASSES[t],CLASSES[p],round(prob*100,2)])

def main():
    """主函数"""
    print(device)
    os.makedirs(f"{output_dir}/model",exist_ok=True)#输出目录先建好，避免保存时报错
    #图像的预处理方法
# 标准 ImageNet 归一化参数
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    eval_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
    ])
    tail = [T.ToTensor(), normalize]
    train_tf = T.Compose([
        T.RandomResizedCrop(224,scale=(0.7,1.0),ratio=(0.75,1.3333)),
        #随机选一块区域（面积是原图 70%-100%），再缩放到224*224，随机缩放+随机位置
        #torchvision 默认是 (0.08, 1.0)（ImageNet 通用值，允许裁到 8%），但蜱虫图主体占图大，裁太小会把虫裁掉 → 下限提到 0.7
        #ratio：允许裁块长宽比在 0.75~1.33 之间微调，保持默认即可
        T.RandomHorizontalFlip(p=0.5),
        #50% 概率左右镜像。蜱虫左右对称（ RandomVerticalFlip：上下颠倒的蜱虫（头朝下）可做）
        T.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        #亮度/对比度/饱和度各 ±20% 随机扰动，模拟不同光照拍摄条件（iNaturalist 照片光照差异极大）
        # - 三个值 0.2 是温和档；调太猛（0.5+）会失真
        *tail
    ])
#实例化数据集
    train_dataset=SplitDataset(
        "train",transforms=train_tf
    )
    val_dataset=SplitDataset(
        "val",transforms=eval_tf
    )
    test_dataset=SplitDataset(
        "test",transforms=eval_tf
    )
    #把预处理的图像张量输入到神经网络
    train_loader=torch.utils.data.DataLoader(
        train_dataset,#要加载的数据集对象（必须是dataset对象）
        batch_size=BATCH_SIZE,#每个batch的样本数
        shuffle=True,#每个epoch是否打乱数据
        num_workers=NUM_WORKERS,#用于数据加载的子进程数
    )

    test_loader=torch.utils.data.DataLoader(
        test_dataset,#要加载的数据集对象（必须是dataset对象）
        batch_size=BATCH_SIZE,#每个batch的样本数
        shuffle=False,#每个epoch是否打乱数据
        num_workers=NUM_WORKERS,#用于数据加载的子进程数
    )
    val_loader=torch.utils.data.DataLoader(
        val_dataset,#要加载的数据集对象（必须是dataset对象）
        batch_size=BATCH_SIZE,#每个batch的样本数
        shuffle=False,#每个epoch是否打乱数据
        num_workers=NUM_WORKERS,#用于数据加载的子进程数
    )
    #加载模型，迁移训练
    model=get_model()

    #损失函数,计算模型预测值与真实标签的差距，作为模型优化目标
    criterion=nn.CrossEntropyLoss()

    #优化器，根据损失函数的计算梯度，更新模型的参数，使损失不断减小，lr(学习率)，weight_decay(权重衰减)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)

    #学习率调度器，在训练过程中动态调整学习率，按照余弦曲线从初始值下降到 0，然后重置。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    #迭代训练每个epoch
    best_f1=0
    for epoch in range(EPOCHS):
        print(f"\n迭代次数：{epoch+1}/{EPOCHS}")
        train_loss,train_acc= train_one_epoch(model,train_loader,criterion,optimizer)
        val_acc, val_f1, _, _, _=evaluate(model,val_loader)
        scheduler.step()#在每个训练周期（epoch）结束时，更新学习率
        print(f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")#每轮指标打印，便于观察收敛
        if (epoch + 1) % 5 == 0:#每 5 个 epoch 保存检查点（包含模型和优化器状态）。
            checkpoint_path = f"{output_dir}/model/tick_species_3cls_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
        if val_f1>best_f1:#用 val F1 选最优（类别不均衡时比 acc 合理）
            best_f1=val_f1
            torch.save(model.state_dict(), output_model_p)
    #评估测试集
    model.load_state_dict(torch.load(output_model_p,weights_only=True))
    #验证集预测 CSV（阈值复现脚本的输入）
    val_acc, val_f1,val_preds, val_labels, val_probs = evaluate(model, val_loader)
    save_val_csv(val_dataset,val_preds,val_labels,val_probs,f"{output_dir}/val_predictions.csv")
    test_acc, test_f1,test_all_preds, test_all_labels, _ = evaluate(model, test_loader)
    plot_confusion_matrix(test_all_preds,test_all_labels,CLASSES,output_dir)
    print(f"测试集评估结果:准确率（Accuracy）:{test_acc: .4f} F1分数：{test_f1:.4f}")
if __name__=="__main__":
    main()