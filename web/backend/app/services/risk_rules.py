"""
风险规则引擎：物种 → 就医建议文案
医学信息依据：蜱虫与蜱传疾病文献（莱姆病/SFTS/森林脑炎/斑疹热/CCHF 等）

设计：规则分两层
1. 精确规则 SPECIES_RULES —— 高关注度物种的专属文案
2. 属级前缀规则 GENUS_RULES —— 同属物种共享风险提示（规则引擎可扩展的体现）
"""
from collections import OrderedDict

# 精确规则：slug -> 提示文案（素材来自 data/tick_species/物种中文名对照表.csv 备注列）
SPECIES_RULES = OrderedDict({
    # 莱姆病/森林脑炎类（硬蜱属重点种）
    "ixodes_scapularis": "肩板硬蜱（黑腿蜱）是莱姆病的主要传播媒介，被叮咬后若出现游走性红斑、发热等症状应及时就医",
    "ixodes_persulcatus": "全沟硬蜱是森林脑炎的主要传播媒介，高发地区被叮咬后建议尽快就医评估",
    "ixodes_ricinus": "篦子硬蜱可传播莱姆病与森林脑炎，注意观察叮咬部位与全身症状",
    # 病毒性出血热类
    "haemaphysalis_longicornis": "长角血蜱是发热伴血小板减少综合征（SFTS）病毒的主要媒介，出现发热、乏力等症状需立即就医",
    "hyalomma_marginatum": "边缘璃眼蜱可传播克里米亚-刚果出血热（CCHF），需高度警惕",
    # 斑疹热类
    "dermacentor_variabilis": "变异革蜱（美国犬蜱）可传播落基山斑疹热，被叮咬后请观察是否出现皮疹、发热",
    "dermacentor_andersoni": "安氏革蜱（落基山木蜱）可传播落基山斑疹热与蜱传麻痹症",
    # 犬类媒介
    "rhipicephalus_sanguineus": "血红扇头蜱可传播犬埃立克体病等立克次体病，宠物家庭需注意",
    # 特殊综合征
    "amblyomma_americanum": "孤星蜱叮咬与 α-gal 综合征（红肉过敏）相关，若进食红肉后出现过敏反应请就医",
})

# 属级前缀规则：按 slug 前缀匹配，兜底同属物种
GENUS_RULES = OrderedDict([
    ("ixodes_", "该硬蜱属物种可能传播莱姆病/蜱传脑炎类疾病，建议就医评估"),
    ("dermacentor_", "该革蜱属物种可能传播蜱传斑疹热类疾病，注意观察发热、皮疹"),
    ("hyalomma_", "该璃眼蜱属物种可能传播病毒性出血热类疾病，请高度警惕并及时就医"),
    ("haemaphysalis_", "该血蜱属物种可能传播 SFTS 等发热伴血小板减少类疾病"),
    ("rhipicephalus_", "该扇头蜱属物种常见于犬类等家畜，可能传播立克次体类疾病"),
    ("amblyomma_", "该花蜱属物种叮咬可能引起过敏反应或立克次体感染"),
])

GENERIC_ADVICE = "建议妥善保存虫体（拍照记录），观察叮咬部位与全身症状，出现异常及时就医"
DISCLAIMER = "本结果仅供辅助参考，不构成医疗诊断建议"

UNKNOWN_ADVICE = "无法确定蜱种，建议拍照留存并寻求专业鉴定"


def advice_for(species_slugs, statuses):
    """汇总风险建议：确认物种的专属文案 + uncertain 提示 + 通用建议 + 免责声明
    参数：
        species_slugs: 检测到的物种 slug 列表（可重复）
        statuses: 与上面对应的决策状态列表（confirmed / uncertain）
    """
    confirmed = []
    for slug, status in zip(species_slugs, statuses):
        if status == "confirmed":
            confirmed.append(slug)

    parts = []
    for slug in dict.fromkeys(confirmed):  # 去重且保持出现顺序
        text = SPECIES_RULES.get(slug)
        if text is None:
            for prefix, rule in GENUS_RULES.items():
                if slug.startswith(prefix):
                    text = rule
                    break
        parts.append(text if text else GENERIC_ADVICE)

    uncertain_count = statuses.count("uncertain")#统计 statuses列表中值为 "uncertain"的元素数量。
    if uncertain_count:
        parts.append(UNKNOWN_ADVICE)

    if not parts:
        parts.append(GENERIC_ADVICE)

    return "；".join(parts) + f"。{DISCLAIMER}。"
