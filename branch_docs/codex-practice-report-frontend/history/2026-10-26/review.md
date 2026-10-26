# 识别重建代码审查

2026-10-26，按requesting-code-review技能执行一次独立只读评审；评审范围dbf87bc1至878f7830及候选级联改动，不启动GPU，不改主分支。审查发现两项P2：

1. 分类训练恢复允许更换out，继承best_score却不迁移原best，可能无best或混入其他实验。当前run01始终原目录恢复，未触发；待正式训练结束后限制原目录，避免中途改变训练源码哈希。
2. 检测训练验证缺少服务IoU=.3去重，会影响重复计数与best选择。已新增重叠框复现测试，先验证失败，再复用服务_dedup修复，进入bda575be。旧pilot数字不重写，正式检测训练须使用修正版本。

实际三任务清单中未发现跨集合group/SHA或正负组冲突；双头损失、采样质量和分类RNG恢复未发现其他具体问题。这不是证明全部数据标签正确，也不代替GPU实测和系统验收。

后续根审查补充：原COCO空标注并非经核验的真负。分类区域清单已不使用这些空图训练存在性；检测正式训练前须复核或隔离，不能只因annotations=[]当作可靠真负。检测pilot仅检查链路，不作为最终效果。

当前40项CPU回归覆盖数据分组、几何增强、mask/box监督、验证完整性、显存修复、EXIF、双头mask loss、采样、校准、级联计数与混合图复核。运行命令为`python -m unittest discover -s tests -q`；GPU仍串行。


## 第二次只读审查

范围：8424cc1d及空mask监督隔离、完整Dataset预检、补充正例筛查、全图检测对照脚本。未运行GPU、未修改训练文件。发现一项P2：profile构建只核对val/cal旧模型哈希彼此一致，未绑定当前baseline文件，可能把旧报告用于改变后的模型/阈值。已修复：绑定baseline文件SHA、species/insect模型SHA、val/cal源清单及coarse guard一致性；回归复现模型、阈值、清单任一不符均拒绝。52项CPU回归通过。其它指定脚本未发现可证实P1/P2。

全图对照的原始新模型框为float、旧服务缓存为int，已在报告明确限制；最终还需真实服务口径复评，不能仅此声明网页验收通过。

## 本轮完成后的复核

独立reviewer继续只读检查v3显式warm-start与恢复身份、全模型SHA记录，未发现新的P1/P2。随后检查VerifiedDetector的原图/tile坐标转换、ROI分数语义及评估缓存身份，未发现确定P1/P2；该架构实际未恢复召回，不进入main。

根agent发现非默认检测门槛没有传入CachedDetector，撤回首个det03_run01报告及profile，补失败→通过回归并完整重跑。reviewer再次检查门槛贯通原生检测、缓存、metadata、profile builder和main.configure_detector，未发现剩余P1/P2；builder拒绝无helper身份的非默认门槛报告和双模型错发布。全部62项CPU回归通过，CUDA HTTP run04与可解码JPEG标注实际通过。没有把代码审查代替模型评估，来源检出代价及9/80未知物种错误确认明确保留。
