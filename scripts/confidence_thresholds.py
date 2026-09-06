"""
复现 TickIDNet 的分物种置信阈值方法（Villon et al., Sci Rep 2020）。

依据：TickIDNet/Confidence Thresholds/confidence_thresholding.py
（Justen et al., "Identification of Public Submitted Tick Images", PLOS ONE 2021）

方法：对每个物种，在 tau ∈ [30, 100] 上搜索决策阈值，约束
"误分类率（MC）< 上界"的同时最大化"正确分类率（CC）"；
置信度低于阈值的样本判为"不确定（unsure，UC）"——宁可不答，不要猜错。

输入 CSV（你的验证集预测结果，三列必有）：
  fname,true,pred,prob
  img1.jpg,Ixodes scapularis,Ixodes scapularis,96.3
  img2.jpg,Ixodes scapularis,Dermacentor variabilis,62.1
（true/pred 可用类名或类别索引；prob 接受 0-100 或 0-1 两种量纲，自动归一）

用法：
  python scripts/confidence_thresholds.py --results results.csv --out output/thresholds.csv
  python scripts/confidence_thresholds.py --demo    # 合成数据演示/自检
"""

import argparse
import random

import numpy as np
import pandas as pd


def load_results(path):
    df = pd.read_csv(path)
    for col in ("fname", "true", "pred", "prob"):
        if col not in df.columns:
            raise ValueError(f"CSV 缺少列 {col!r}，当前列: {list(df.columns)}")
    df = df.dropna(subset=["true", "pred", "prob"])
    if df["prob"].max() <= 1.5:  # 0-1 量纲 -> 百分比
        df["prob"] = df["prob"] * 100
    return df


def threshold_for_class(df, tau, mc_bound):
    """
    单个类别的阈值搜索。df 须含 prob / true / pred 且只含该类别样本。
    返回 (threshold, cc, mc, uc, final_acc_excl_unsure)
    """
    n = len(df)
    cc = np.array([(df.loc[(df["prob"] > t) & (df["pred"] == df["true"])].shape[0]) / n
                   for t in tau])
    mc = np.array([(df.loc[(df["prob"] > t) & (df["pred"] != df["true"])].shape[0]) / n
                   for t in tau])
    uc = np.array([(df.loc[df["prob"] < t]).shape[0] / n for t in tau])

    # 与原文一致：t=100 的点不参与"mc < bound"的选择（该点恒为 cc=0）
    valid = mc[:-1] < mc_bound
    if valid.any():
        # 在满足 mc < bound 的阈值中选 cc 最大者（等价于原文 goal_two）
        idx = np.where(valid)[0][np.argmax(cc[:-1][valid])]
    else:
        # 没有阈值能把 mc 压到 bound 以下：取 mc 最小的阈值
        idx = np.argmin(mc[:-1])
    return tau[idx], cc[idx], mc[idx], uc[idx], cc[idx] / (1 - uc[idx])


def compute_thresholds(df, bounds=(10, 5, 2)):
    tau = np.linspace(30, 100, num=1000)
    classes = sorted(df["true"].unique())
    rows, thresholds = [], []
    for cls in classes:
        sub = df[df["true"] == cls]
        n = len(sub)
        row = {"类别": cls, "样本数": n}
        cc_none = (sub["pred"] == sub["true"]).mean() * 100
        mc_none = (sub["pred"] != sub["true"]).mean() * 100
        row["无阈值 CC%"] = round(cc_none, 2)
        row["无阈值 MC%"] = round(mc_none, 2)
        for b in bounds:
            t, cc, mc, uc, acc = threshold_for_class(sub, tau, b / 100)
            row[f"MC<{b}% 阈值"] = round(float(t), 1)
            row[f"MC<{b}% CC%"] = round(cc * 100, 2)
            row[f"MC<{b}% UC%"] = round(uc * 100, 2)
            row[f"MC<{b}% 排除不确定后准确率%"] = round(acc * 100, 2)
        thresholds.append(row)
    return pd.DataFrame(thresholds)


def demo(n_per_class=200):
    """合成演示数据：三类，难度仿 TickIDNet（孤星蜱最难分）"""
    random.seed(0)
    np.random.seed(0)
    spec = {  # 类别: (正确置信度均值/标准差, 错判置信度均值/标准差, 正确率)
        "Amblyomma americanum": (82, 12, 0.78),
        "Dermacentor variabilis": (90, 9, 0.91),
        "Ixodes scapularis": (87, 11, 0.87),
    }
    rows = []
    classes = list(spec)
    for i, cls in enumerate(classes):
        mu, sd, acc = spec[cls]
        for j in range(n_per_class):
            if random.random() < acc:
                pred, p = cls, np.random.normal(mu, sd)
            else:
                pred = random.choice([c for c in classes if c != cls])
                p = np.random.normal(mu - 20, sd)
            rows.append({"fname": f"{cls[:3]}_{j}.jpg", "true": cls,
                         "pred": pred, "prob": np.clip(p, 1, 99)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="验证集预测结果 CSV（fname,true,pred,prob）")
    ap.add_argument("--out", default="output/thresholds.csv")
    ap.add_argument("--demo", action="store_true", help="合成数据自检")
    args = ap.parse_args()

    df = demo() if args.demo else load_results(args.results)
    table = compute_thresholds(df)
    print(table.to_string(index=False))
    if not args.demo:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n阈值表已保存 -> {args.out}")


if __name__ == "__main__":
    main()
