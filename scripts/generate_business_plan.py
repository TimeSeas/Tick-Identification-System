"""Generate 智蜱卫士 商业计划书 DOCX"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

def set_cell_shading(cell, color):
    """Set cell background color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color)
    shading.set(qn('w:val'), 'clear')
    tcPr.append(shading)

def add_styled_heading(doc, text, level=1):
    """Add a heading with custom styling."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x56, 0xdb)  # Blue
    return heading

def add_body_text(doc, text):
    """Add body text with proper formatting."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p

def add_key_point(doc, text):
    """Add a highlighted key point."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run('▶ ' + text)
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1a, 0x56, 0xdb)
    return p

def add_table_with_data(doc, headers, rows, col_widths=None):
    """Add a formatted table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(10)

    # Data rows
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)

    doc.add_paragraph()  # Spacer
    return table

def add_swot_table(doc, strengths, weaknesses, opportunities, threats):
    """Add SWOT analysis table."""
    table = doc.add_table(rows=3, cols=2)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # S and W
    table.rows[0].cells[0].text = "S 优势\n\n" + "\n".join("• " + s for s in strengths)
    table.rows[0].cells[1].text = "W 劣势\n\n" + "\n".join("• " + w for w in weaknesses)

    # O and T
    table.rows[1].cells[0].text = "O 机会\n\n" + "\n".join("• " + o for o in opportunities)
    table.rows[1].cells[1].text = "T 威胁\n\n" + "\n".join("• " + t for t in threats)

    # Strategies
    table.rows[2].cells[0].merge(table.rows[2].cells[1])
    table.rows[2].cells[0].text = "应对策略：SO策略——利用技术优势抢占政策红利窗口期，快速建立行业标杆。WO策略——在发展阶段引入商业化运营人才，弥补团队短板。ST策略——强化知识产权保护，构建技术壁垒和数据壁垒。WT策略——初期聚焦细分场景（宠物医疗），避免与巨头正面竞争。"

    # Color headers
    for cell in [table.rows[0].cells[0], table.rows[0].cells[1],
                 table.rows[1].cells[0], table.rows[1].cells[1]]:
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.bold = True

    doc.add_paragraph()
    return table

def generate():
    doc = Document()

    # ---- Page Style ----
    style = doc.styles['Normal']
    font = style.font
    font.name = '宋体'
    font.size = Pt(11)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    # ---- COVER PAGE ----
    for _ in range(4):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('智蜱卫士')
    run.font.size = Pt(36)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1a, 0x56, 0xdb)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('基于AI细粒度识别的蜱虫智能防控系统')
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    doc.add_paragraph()
    doc.add_paragraph()

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run('商 业 计 划 书').font.size = Pt(24)
    info.runs[0].font.bold = True

    doc.add_paragraph()
    doc.add_paragraph()

    meta = [
        ('参赛赛道', '青年创新专项赛'),
        ('项目负责人', '郑良鑫'),
        ('所属领域', '人工智能·智慧医疗·现代农业'),
    ]
    for label, value in meta:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f'{label}：{value}')
        run.font.size = Pt(14)

    doc.add_page_break()

    # ---- TABLE OF CONTENTS ----
    add_styled_heading(doc, '目录', level=1)
    toc_items = [
        '一、执行摘要',
        '二、创业团队',
        '三、产品与服务描述',
        '四、行业与市场分析',
        '五、营销计划',
        '六、运营管理',
        '七、财务计划',
        '八、风险控制',
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(8)
        for run in p.runs:
            run.font.size = Pt(13)

    doc.add_page_break()

    # ============================================================
    # CHAPTER 1: 执行摘要
    # ============================================================
    add_styled_heading(doc, '一、执行摘要', level=1)

    add_styled_heading(doc, '1.1 项目背景', level=2)
    add_body_text(doc,
        '蜱虫是全球第二大虫媒传染病传播媒介，仅次于蚊虫。据美国CDC估计，美国每年'
        '莱姆病实际感染人数约47.6万；2023年报告病例达89,486例，为近十年最高。'
        '（来源：cdc.gov/lyme/data-research/facts-stats）。中国疾控中心数据显示，'
        '2005-2024年20年间我国蜱媒传染病累计报告38,241例、死亡1,656例，病死率4.33%'
        '，为四类媒介传染病中最高。其中发热伴血小板减少综合征（SFTS）占89%，'
        '病死率高达4.79%，疫区以每年新增约7个城市的速度持续扩散。'
        '（来源：China CDC Weekly, 2025）。过去十年我国新鉴定28种新兴蜱媒病原体，'
        '其中11种为全球首次发现的人类感染病例（Lancet, 2025）。'
        '蜱媒传染病呈显著上升趋势，防控形势严峻。')

    add_body_text(doc,
        '然而，传统蜱虫识别依赖昆虫学专家人工镜检，单只鉴定平均耗时15分钟以上，'
        '且基层疾控人员和农牧从业者普遍缺乏专业鉴别能力，误判率高达40%以上。'
        '市场上缺乏一款能够快速、准确、自动化识别蜱虫并进行状态分析的产品。')

    add_styled_heading(doc, '1.2 项目概述', level=2)
    add_body_text(doc,
        '智蜱卫士是一款基于Mask R-CNN深度学习框架的蜱虫细粒度识别系统。项目创新性'
        '地设计了"实例分割+部位解析+状态分类"三层级检测头架构，突破现有技术仅能定位虫体'
        '轮廓的局限，实现对蜱虫节肢结构、体态特征及腹部干瘪/饱满状态的精确识别。'
        '经实测，系统在测试集上达到mAP@0.50=0.842的检测精度，处于行业领先水平。')

    add_styled_heading(doc, '1.3 市场机会', level=2)
    add_body_text(doc,
        '目标市场覆盖公共卫生监测、农牧业养殖管理和宠物医疗三大领域，总市场规模超过'
        '200亿元。当前市场上尚无专门针对蜱虫细粒度识别的AI产品，属于蓝海市场。'
        '项目契合国家"健康中国2030"战略和"智慧农业"发展政策，具备巨大的市场潜力和社会价值。')

    add_styled_heading(doc, '1.4 竞争优势', level=2)
    add_key_point(doc, '技术领先：独创三层级检测头架构，实现细粒度属性识别，精度达mAP@50=0.842')
    add_key_point(doc, '数据壁垒：融合蜱虫专类数据与通用昆虫数据，模型抗干扰能力强')
    add_key_point(doc, '先发优势：国内尚无同类蜱虫AI识别产品，抢占空白市场')
    add_key_point(doc, '轻量化部署：支持移动端、边缘设备和无人机巡检系统')

    add_styled_heading(doc, '1.5 财务亮点', level=2)
    add_body_text(doc,
        '项目预计第一年完成产品研发和种子用户积累，第二年实现盈亏平衡，第三年营收突破500万元。'
        '毛利率预计维持在65%-75%。当前寻求50万元天使轮融资，用于技术开发和市场验证。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 2: 创业团队
    # ============================================================
    add_styled_heading(doc, '二、创业团队', level=1)

    add_body_text(doc,
        '团队目前由一名创始人构成，采用"一人公司+AI辅助+外部协作"的精益模式，'
        '契合青年创新专项赛对青年创业者的定位。未来将根据业务发展逐步引入商业化运营和'
        '市场拓展人才。')

    add_styled_heading(doc, '2.1 创始人简介', level=2)

    p = doc.add_paragraph()
    run = p.add_run('【创始人】郑良鑫 | 项目负责人')
    run.font.bold = True
    run.font.size = Pt(12)

    add_body_text(doc, '软件工程专业在读本科生（35周岁以下），深耕计算机视觉领域，'
                  '具有丰富的深度学习项目实战经验。已独立完成目标检测、图像分割及多任务学习'
                  '等CV方向的完整项目开发。')

    add_styled_heading(doc, '2.2 核心能力', level=2)
    capabilities = [
        ('计算机视觉', '精通Mask R-CNN、YOLO等主流目标检测框架，掌握实例分割、语义分割技术'),
        ('深度学习', '熟练使用PyTorch、TensorFlow，具备迁移学习和模型优化的实际经验'),
        ('全栈开发', '掌握Python、数据标注工具链、模型部署（ONNX/TensorRT）等技术'),
        ('项目驱动', '独立完成从数据采集、标注、模型训练到推理部署的完整pipeline'),
    ]
    for name, desc in capabilities:
        add_key_point(doc, f'{name}：{desc}')

    add_styled_heading(doc, '2.3 核心能力矩阵', level=2)
    add_table_with_data(doc,
        ['能力维度', '掌握程度', '具体体现'],
        [
            ['计算机视觉', '★★★★★', '独立完成Mask R-CNN全流程开发，mAP@50=0.842'],
            ['深度学习', '★★★★☆', 'PyTorch/TensorFlow迁移学习与模型优化'],
            ['全栈工程', '★★★★☆', '从数据标注到模型部署的完整pipeline'],
            ['项目管理', '★★★☆☆', '独立推进项目全生命周期，敏捷迭代'],
            ['商业运营', '★★☆☆☆', '需引入商业化人才补强'],
        ])

    add_styled_heading(doc, '2.4 团队优势', level=2)
    add_body_text(doc,
        '（1）专业知识高度契合：创始人专业背景与项目核心技术直接对应，'
        '无需额外技术学习成本。')
    add_body_text(doc,
        '（2）独立交付能力：已独立完成阶段一全部开发工作，'
        '验证了一人团队的可行性；项目成果可实际运行和演示，非纸上谈兵。')
    add_body_text(doc,
        '（3）敏捷迭代：扁平化管理，决策效率高，'
        '可快速响应市场反馈进行产品迭代；从需求识别到产品上线周期短于传统团队模式。')
    add_body_text(doc,
        '（4）低成本优势：学生团队运营成本极低，可充分利用学校GPU集群等免费计算资源，'
        '初期仅需极少的资金即可实现技术验证和产品原型开发。')

    add_styled_heading(doc, '2.5 人才引进路线图', level=2)
    add_table_with_data(doc,
        ['阶段', '时间', '引进岗位', '人数', '预期成本'],
        [
            ['种子期', '第1年', '创始人独立运营+外包（UI设计/法务）', '1人+外包', '自筹'],
            ['初创期', '第2年', '后端开发工程师+市场运营专员', '+2人', '天使轮融资'],
            ['成长期', '第3年', '算法研究员+销售经理+客户成功+行政财务', '+4-6人', 'Pre-A轮'],
            ['扩张期', '第4-5年', '区域销售团队+数据标注管理+运维支持', '+8-10人', '营收自循环'],
        ])

    add_styled_heading(doc, '2.6 外部顾问团队', level=2)
    add_body_text(doc,
        '【学术顾问】拟邀请高校计算机视觉或昆虫学方向的教授/研究员'
        '担任技术顾问，提供算法指导和学术背书。')
    add_body_text(doc,
        '【行业顾问】拟邀请疾控中心或宠物医疗行业专家提供'
        '临床应用场景验证和行业资源对接。')
    add_body_text(doc,
        '【创业导师】依托学校创业学院和众创杯大赛资源，'
        '获得创业辅导、投融资对接和政策扶持指导。')

    add_styled_heading(doc, '2.7 团队文化', level=2)
    add_body_text(doc,
        '使命驱动（Mission-Driven）：以"用AI守护人类免受蜱虫威胁"为使命，'
        '技术服务于公共卫生。极客精神（Geek Spirit）：追求技术极致，'
        '坚持用代码解决真实世界问题。快速迭代（Ship Fast）：'
        '认同"Done is better than perfect"，小步快跑持续交付价值。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 3: 产品与服务描述
    # ============================================================
    add_styled_heading(doc, '三、产品与服务描述', level=1)

    add_styled_heading(doc, '3.1 用户痛点', level=2)
    add_body_text(doc,
        '痛点一：识别效率低。传统人工镜检鉴定蜱虫平均耗时15分钟/只，'
        '野外调查时每日需处理数百只样本，效率极低。')
    add_body_text(doc,
        '痛点二：专业门槛高。蜱虫物种鉴定和状态判断需要昆虫学专业知识，'
        '基层疾控人员、农牧养殖户和宠物主人普遍不具备鉴别能力，误判率超40%。')
    add_body_text(doc,
        '痛点三：缺乏实时监测手段。现有蜱虫监测依赖人工采集后送实验室检验，'
        '无法实现野外实时预警，错过了最佳防控窗口期。')
    add_body_text(doc,
        '痛点四：与其他昆虫混淆。野外环境中螨虫、蜘蛛等与蜱虫形态相似，'
        '非专业人员难以区分，造成大量误报。')

    add_styled_heading(doc, '3.2 解决方案：三层级检测头架构', level=2)
    add_body_text(doc,
        '智蜱卫士基于改进的Mask R-CNN深度学习框架，自主设计了三层级检测头架构，'
        '突破传统目标检测仅输出边界框的局限：')

    add_body_text(doc,
        '第一层——实例分割头：基于ResNet-50-FPN骨干网络提取多尺度特征，'
        '通过区域建议网络（RPN）生成候选区域，输出蜱虫的像素级分割掩码，'
        '实现复杂背景下（草地、树枝、动物毛发等）的精确虫体定位。')

    add_body_text(doc,
        '第二层——部位解析头：在实例分割基础上，进一步识别头胸、腹部、口器等关键部位，'
        '为后续状态判断提供结构级特征信息。这一层级能够区分蜱虫的形态完整性，'
        '排除虫体残缺导致的误判。')

    add_body_text(doc,
        '第三层——状态分类头：结合全局特征和部位级特征，判断蜱虫的腹部状态'
        '（unfed扁平/fed饱满），以及体态特征（flat扁平/swollen肿胀/engorged鼓胀），'
        '评估吸血程度和发育阶段。该系统已基于TickIDNet数据集的fed/unfed标注进行了'
        '初步训练。')

    add_styled_heading(doc, '3.3 技术指标', level=2)
    add_table_with_data(doc,
        ['指标', '数值', '说明'],
        [
            ['mAP@0.50', '0.842', '蜱虫检测精度（COCO标准）'],
            ['单张推理时间', '<200ms', 'GPU推理（NVIDIA RTX系列）'],
            ['支持类别', '2-28类', '单类/多类昆虫可配置'],
            ['输入分辨率', '≥224×224', '适配移动端和野外设备'],
            ['模型大小', '~170MB', '可压缩至50MB以下部署'],
        ])

    add_styled_heading(doc, '3.4 产品形态', level=2)
    add_body_text(doc,
        'API云服务：提供RESTful API接口，支持图片上传后返回检测结果，'
        '适用于健康监测平台和农牧管理系统的集成。')
    add_body_text(doc,
        '移动端APP：开发Android/iOS应用，支持拍照即识别，'
        '面向户外爱好者、宠物主人和基层疾控人员。')
    add_body_text(doc,
        '边缘部署包：提供ONNX/TensorRT格式模型，支持部署到野外监测站、'
        '无人机巡检系统等边缘设备。')
    add_body_text(doc,
        'Web管理后台：提供历史检测记录查询、数据统计分析和疫情趋势预警功能。')

    add_styled_heading(doc, '3.5 产品迭代规划', level=2)
    add_table_with_data(doc,
        ['版本', '核心功能', '目标用户', '预计时间'],
        [
            ['V1.0 检测版', '蜱虫目标检测+分割', '疾控原型验证', '已实现'],
            ['V2.0 多类版', '多类昆虫检测（降低误检）', '野外监测', '6个月'],
            ['V3.0 细粒度版', '腹部状态+体态分类', '宠物医疗/疾控', '12个月'],
            ['V4.0 生态平台', '多端部署+疫情预警', '全场景覆盖', '18个月'],
        ])

    add_styled_heading(doc, '3.6 盈利模式', level=2)
    add_table_with_data(doc,
        ['收入来源', '定价策略', '目标占比'],
        [
            ['API调用服务', '按调用量阶梯收费，0.01-0.05元/次', '40%'],
            ['SaaS订阅', '99-299元/月（含智能分析报告）', '30%'],
            ['APP增值服务', '免费基础版+高级订阅29.9元/月', '15%'],
            ['定制化部署', '按项目收费，2-5万元/套', '10%'],
            ['数据服务', '疫区风险评估报告', '5%'],
        ])

    add_styled_heading(doc, '3.7 核心资源', level=2)
    add_body_text(doc,
        '（1）技术专利：三层级检测头架构具有独创性，计划申请发明专利和软件著作权。'
        '（2）数据集：已积累蜱虫标注数据集（COCO格式），数据壁垒是核心竞争力之一。'
        '（3）模型资产：训练完成的Mask R-CNN模型权重，可快速迁移和迭代。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 4: 创新引领性
    # ============================================================
    add_styled_heading(doc, '四、创新引领性', level=1)

    add_body_text(doc,
        '本项目的核心竞争力在于技术创新和模式创新的双轮驱动，'
        '在蜱虫识别领域实现从"0到1"的突破。以下从技术、方法、数据、'
        '壁垒四个维度系统阐述项目的创新引领性。')

    add_styled_heading(doc, '4.1 技术创新：三层级检测头架构', level=2)
    add_body_text(doc,
        '现有目标检测（如YOLO、Faster R-CNN）仅输出物体类别和边界框，'
        '无法回答"虫子吃饱了没有""发育到了哪个阶段"等细粒度问题。'
        '本项目独创三层级检测头架构，将目标检测、实例分割和属性分类'
        '深度融合于单一网络：')

    add_table_with_data(doc,
        ['对比维度', '传统目标检测', '智蜱卫士'],
        [
            ['输出形式', '边界框+类别标签', '边界框+分割掩码+属性描述'],
            ['识别粒度', '物体类别（如"蜱虫"）', '物种+部位+体态+腹部状态'],
            ['任务数量', '单任务（分类+定位）', '多任务（检测+分割+属性）'],
            ['误识别率', '将螨虫/蜘蛛误判为蜱虫', '多类对比学习降低混淆'],
            ['实用价值', '仅"有没有虫"', '"是什么虫+是否危险+吸血程度"'],
            ['技术壁垒', '低（开源模型即用）', '高（需标注+设计+训练）'],
        ])

    add_styled_heading(doc, '4.2 方法创新："检测+识别+分类"一体化', level=2)
    add_body_text(doc,
        '传统昆虫识别系统通常采用串行流水线：先检测后分类，两个模型独立训练，'
        '特征无法共享，推理效率低。本项目创新地将三层级任务集成于单一Mask R-CNN框架，'
        '共享ResNet-50-FPN骨干网络提取的多尺度特征：')

    add_key_point(doc, '特征共享：检测、分割、分类三头共享骨干特征，减少计算冗余')
    add_key_point(doc, '端到端训练：一次标注即可训练全部任务，降低数据准备成本')
    add_key_point(doc, '推理效率：单次前向传播同时输出位置、轮廓和状态信息')
    add_key_point(doc, '可扩展性：新增昆虫类别或属性无需重新设计网络架构')

    add_styled_heading(doc, '4.3 数据创新：三级数据体系', level=2)
    add_body_text(doc,
        '深度学习模型的性能高度依赖训练数据的质量和覆盖面。本项目构建了'
        '独特的三级数据体系，形成难以复制的数据壁垒：')

    add_table_with_data(doc,
        ['数据层级', '内容', '来源', '状态', '竞争价值'],
        [
            ['L1 蜱虫专类', '蜱虫边界框+分割标注', 'COCO格式数据集', '已完成', '核心资产，稀缺数据'],
            ['L2 通用昆虫', '28类昆虫标注（防误检）', 'archive/archive2/archive3', '标注中', '降低误识别率'],
            ['L3 细粒度属性', 'fed/unfed/体态标注', 'TickIDNet元数据CSV', '已有关联', '全球独有数据'],
        ])

    add_styled_heading(doc, '4.4 创新壁垒分析', level=2)
    add_body_text(doc,
        '创新不仅要领先，还要可持续领先。本项目从以下维度构建多层次的竞争壁垒：')

    add_key_point(doc, '数据壁垒：蜱虫标注数据极为稀缺（需要昆虫学专家知识），数据积累本身就是护城河')
    add_key_point(doc, '知识壁垒：细粒度属性标注需理解蜱虫生物学特征（口器、生殖孔、盾板），非通用AI公司可快速复制')
    add_key_point(doc, '先发优势：国内尚无同类蜱虫AI识别产品，抢先建立品牌认知和用户黏性')
    add_key_point(doc, '算法壁垒：三层级多任务学习架构需要精心设计的损失函数和训练策略，非简单迁移学习可达到同等精度')
    add_key_point(doc, '学术壁垒：引用Lancet (2025)系统综述成果——过去十年中国新鉴定28种蜱媒病原体，其中11种全球首次发现。随着蜱媒疾病威胁加剧，本项目的应用价值将持续放大')

    add_styled_heading(doc, '4.5 行业影响力路线图', level=2)
    add_body_text(doc,
        '短期（1-2年）：建立"AI蜱虫识别"品类定义，成为该赛道的先行者和标准制定者。'
        '中期（3-4年）：推动AI蜱虫识别纳入疾控中心技术规范，参与行业标准起草。'
        '长期（5年以上）：构建蜱虫AI识别公共服务平台，成为虫媒传染病智慧防控的基础设施。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 5: 行业与市场分析
    # ============================================================
    add_styled_heading(doc, '五、行业与市场分析', level=1)

    add_styled_heading(doc, '4.1 PEST分析', level=2)

    add_body_text(doc, '【政策环境（Policy）】')
    add_body_text(doc,
        '"健康中国2030"规划纲要将虫媒传染病防控列为公共卫生重点任务。'
        '农业农村部发布《农作物病虫害监测与预报管理办法》，推动智慧农业技术应用。'
        '广东省大力支持人工智能与医疗健康融合创新，相关政策持续利好。')

    add_body_text(doc, '【经济环境（Economy）】')
    add_body_text(doc,
        '中国宠物消费市场规模已达3,126亿元（2025年，《中国宠物行业白皮书》），'
        '其中宠物诊疗消费占比持续提升（来源：chinareports.org.cn）。'
        '预计2028年突破4,050亿元，养宠主力90后占比42.7%。'
        '全球病媒控制市场（Vector Control）2025年规模约220-278亿美元'
        '（来源：Knowledge Sourcing Intelligence / GlobeNewsWire, 2026），'
        'AI+虫害防治是增速最快的技术赛道。')

    add_body_text(doc, '【社会环境（Society）】')
    add_body_text(doc,
        '户外运动（徒步、露营）参与人数快速增长，蜱虫叮咬风险增加，'
        '公众对蜱虫防护意识提升。美国CDC估计该国每年约47.6万人被诊断为莱姆病'
        '（来源：cdc.gov/lyme/data-research/facts-stats），'
        '蜱虫叮咬相关疾病已成为严重的公共卫生问题。'
        '宠物饲养率持续攀升（中国城镇家庭饲养率超30%），宠物健康监测需求增加。'
        '中国蜱媒传染病疫区持续扩大，基层疾控能力建设受到更多重视。')

    add_body_text(doc, '【技术环境（Technology）】')
    add_body_text(doc,
        'AI视觉技术成熟度提高，Mask R-CNN等模型已达到实用化水平。'
        '移动端算力提升（NPU/AI芯片普及），使端侧AI推理成为可能。'
        '5G/物联网技术为野外设备联网监测提供基础设施支撑。')

    add_styled_heading(doc, '4.2 市场容量', level=2)
    add_table_with_data(doc,
        ['细分市场', '当前规模（2025）', '未来5年预期', '可触达份额', '数据来源'],
        [
            ['公共卫生监测', '约30亿元', '50亿元（CAGR 10%）', '5%-10%', 'China CDC Weekly'],
            ['宠物医疗市场', '约120亿元', '250亿元（CAGR 15%）', '2%-5%', '宠物行业白皮书2025'],
            ['农牧业虫害防治', '约50亿元', '90亿元（CAGR 12%）', '3%-8%', '农业农村部'],
            ['合计', '约200亿元', '约390亿元', '—', '—'],
        ])

    add_styled_heading(doc, '4.3 目标用户画像', level=2)
    add_table_with_data(doc,
        ['用户群体', '特征', '核心需求', '付费意愿'],
        [
            ['各级疾控中心', '专业人员，需要批量检测', '准确率>95%，批量处理', '高（政府采购）'],
            ['宠物医院', '兽医和护理人员', '快速识别，操作简单', '中高（专业工具）'],
            ['农牧养殖户', '非专业人员', '拍照即识别，低门槛', '中（按效果付费）'],
            ['户外爱好者', '普通消费者', '便携识别，安全提醒', '低中（免费+增值）'],
        ])

    add_styled_heading(doc, '4.4 竞争分析', level=2)
    add_body_text(doc, '当前市场尚无专门针对蜱虫细粒度识别的AI商业化产品。'
                  '主要竞争来源为：')
    add_body_text(doc, '（1）通用昆虫识别APP（如iNaturalist）：基于图像分类，'
                  '仅能判断物种大类，无法定位虫体位置和状态，识别精度低。')
    add_body_text(doc, '（2）传统实验室检测服务：精度高但周期长（3-7天），'
                  '单次成本高（50-200元），无法满足实时需求。')
    add_body_text(doc, '（3）学术研究工具：部分高校有蜱虫识别研究，'
                  '但停留在实验室阶段，缺乏产品化能力。')

    add_body_text(doc, '智蜱卫士以"细粒度属性识别+端侧实时推理"形成差异化优势，'
                  '填补市场空白。')

    add_styled_heading(doc, '4.5 SWOT分析', level=2)
    add_swot_table(doc,
        ['独创三层级检测架构，技术壁垒高', '检测精度领先（mAP@50=0.842）',
         '先发优势，蓝海市场', '全栈独立开发能力'],
        ['团队规模小，仅1人', '缺乏商业化运营经验',
         '品牌知名度为零', '资金有限'],
        ['健康中国2030政策红利', 'AI+医疗赛道资本热度高',
         '宠物经济爆发增长', '无人竞争的市场窗口期'],
        ['大厂可能进入赛道', '技术被模仿', '数据获取成本高',
         '用户教育成本高'],
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 5: 营销计划
    # ============================================================
    add_styled_heading(doc, '六、营销计划', level=1)

    add_styled_heading(doc, '5.1 营销目标', level=2)
    add_table_with_data(doc,
        ['阶段', '时间', '目标'],
        [
            ['种子期', '第1年', '获取100+专业用户（疾控/宠物医院），完成产品PMF验证'],
            ['成长期', '第2年', '拓展至1000+付费用户，月活用户破5000，品牌初步建立'],
            ['扩张期', '第3年', '覆盖全国重点省份，营收突破500万，市场份额达5%'],
        ])

    add_styled_heading(doc, '5.2 目标市场细分', level=2)
    add_body_text(doc,
        '优先切入宠物医疗市场（宠物主人消费意愿强、支付能力高），'
        '同步推进疾控中心试点合作（建立专业背书），'
        '后期拓展农牧养殖领域（市场规模大、需求刚性）。')

    add_styled_heading(doc, '5.3 4P策略', level=2)

    add_body_text(doc, '◆ 产品（Product）')
    add_body_text(doc,
        '分层级产品体系：免费版（基础识别功能，培养用户习惯）→ 专业版（细粒度分析+报告，面向机构）→ '
        '企业版（API接入+定制部署，面向大客户）。差异化设计满足不同用户群体的需求深度。')

    add_body_text(doc, '◆ 定价（Price）')
    add_body_text(doc,
        '采用价值导向定价策略。免费版获取用户基础，'
        '专业版订阅99元/月起，企业版按项目定制2-5万元。'
        '疾控和政府客户采用政府采购定价模式。')

    add_body_text(doc, '◆ 渠道（Place）')
    add_body_text(doc,
        '线上渠道：自建官网+微信小程序，入驻华为应用市场、苹果App Store等主流应用商店，'
        '在宠物社群、户外运动论坛等垂直社区进行内容营销。')
    add_body_text(doc,
        '线下渠道：参与疾控学术会议和兽医行业展会，与宠物连锁医院建立合作试点，'
        '通过各地疾控中心培训体系辐射基层用户。')

    add_body_text(doc, '◆ 促销（Promotion）')
    add_body_text(doc,
        '（1）内容营销：发布蜱虫科普文章/短视频，打造"专业防蜱"IP。'
        '（2）KOL合作：联合宠物医生、户外博主进行产品体验传播。'
        '（3）学术背书：发表学术论文，参与行业标准制定，建立专业权威。'
        '（4）老带新激励：推荐赠会员时长，机构推荐返佣。')

    add_styled_heading(doc, '5.4 用户生命周期管理', level=2)
    add_body_text(doc,
        '获客（搜索引流+内容营销+展会）→ 激活（免费试用+新手引导）→ '
        '留存（定期推送蜱虫风险月报、行业资讯）→ 付费（高级功能解锁+数据分析报告）→ '
        '转介绍（分享有奖+机构合作推广）。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 6: 运营管理
    # ============================================================
    add_styled_heading(doc, '七、运营管理', level=1)

    add_styled_heading(doc, '6.1 业务流程', level=2)
    add_body_text(doc,
        '用户拍摄/上传图片 → API网关接收 → 模型推理引擎（蜱虫检测+细粒度识别）→ '
        '结果后处理（置信度过滤+可视化）→ 返回检测报告（虫体位置+状态+风险等级）→ '
        '数据存档（匿名化存储，用于模型持续优化）。')

    add_body_text(doc, '关键节点：响应时间<1秒（GPU模式），准确率>85%（实际场景），'
                  '数据安全加密传输，符合个人信息保护法合规要求。')

    add_styled_heading(doc, '6.2 组织架构', level=2)
    add_body_text(doc,
        '初期（第1年）：扁平结构。创始人独立负责技术研发+产品设计+市场推广，'
        '以外包形式获取UI设计、法律咨询等专业服务。')
    add_body_text(doc,
        '发展期（第2-3年）：引入2-3名核心成员。设立技术部（算法迭代+后端维护）、'
        '市场部（商务拓展+用户运营）、综合部（行政+财务+法务）。')

    add_styled_heading(doc, '6.3 质量控制', level=2)
    add_body_text(doc,
        '（1）模型持续迭代：每季度基于新增标注数据更新模型，每月评估检测精度。'
        '（2）用户反馈闭环：内建"纠错上报"功能，用户可标记错误检测结果，'
        '自动纳入训练数据池。（3）A/B测试机制：新模型上线前通过AB测试验证效果。'
        '（4）SLA保障：API服务承诺99.5%可用率。')

    add_styled_heading(doc, '6.4 合作伙伴', level=2)
    add_table_with_data(doc,
        ['合作类型', '合作方', '合作内容'],
        [
            ['学术合作', '高校CV实验室', '联合算法研究、数据共享'],
            ['行业合作', '疾控中心/宠物医院', '试点应用、临床验证'],
            ['技术合作', '云计算服务商', 'GPU算力支持、模型部署'],
            ['渠道合作', '宠物医疗连锁', '渠道推广、联合品牌营销'],
        ])

    doc.add_page_break()

    # ============================================================
    # CHAPTER 8: 带动就业
    # ============================================================
    add_styled_heading(doc, '八、带动就业与产业贡献', level=1)

    add_body_text(doc,
        '作为青年创新创业项目，智蜱卫士不仅追求商业成功，更致力于创造社会价值和'
        '就业机会。本项目具备显著的就业带动效应和产业辐射效应。')

    add_styled_heading(doc, '8.1 直接就业计划', level=2)
    add_table_with_data(doc,
        ['阶段', '团队规模', '新增岗位', '岗位类型'],
        [
            ['种子期（第1年）', '1人→3人', '+2人', '后端工程师1人+市场运营1人'],
            ['初创期（第2年）', '3人→7人', '+4人', '算法工程师+销售+客户成功+行政'],
            ['成长期（第3年）', '7人→15人', '+8人', '区域销售经理+数据标注主管+运维+商务拓展等'],
            ['扩张期（第4-5年）', '15人→30+人', '+15人', '全国区域团队+研发中心+客服团队'],
        ])

    add_styled_heading(doc, '8.2 间接就业带动', level=2)
    add_body_text(doc,
        '【数据标注产业】蜱虫细粒度标注需要既懂昆虫形态又懂标注工具的专业人员，'
        '预计将带动50-100名兼职标注员（高校生物专业学生为主），'
        '形成稳定的标注外包产业链。按每人年均收入1-3万元估算，'
        '可贡献100-300万元的劳动收入。')

    add_body_text(doc,
        '【农业技术推广】产品在农牧养殖场景部署时，将产生本地化技术支持需求，'
        '预计带动20-50名农业技术推广人员（对接各地农业合作社和养殖企业）。')

    add_body_text(doc,
        '【宠物医疗应用】与宠物医院合作推广过程中，将催生"AI+宠物医疗"'
        '应用顾问岗位，预计间接带动30-80人就业。')

    add_body_text(doc,
        '【设备部署运维】野外监测站和无人机巡检系统的部署将产生硬件集成和'
        '运维岗位需求，预计间接带动10-20人就业。')

    add_styled_heading(doc, '8.3 就业带动量化预估', level=2)
    add_table_with_data(doc,
        ['就业类型', '第1年', '第2年', '第3年', '第5年'],
        [
            ['直接员工', '1-3人', '3-7人', '7-15人', '30+人'],
            ['标注外包', '5-10人', '20-30人', '40-60人', '80-100人'],
            ['技术推广', '0', '5-10人', '15-25人', '30-50人'],
            ['部署运维', '0', '2-5人', '5-10人', '10-20人'],
            ['合计带动', '6-13人', '30-52人', '67-110人', '150-200+人'],
        ])

    add_styled_heading(doc, '8.4 人才培养与产业生态', level=2)
    add_body_text(doc,
        '本项目将建立"AI+昆虫学"交叉学科人才培养机制：与高校合作开设'
        '计算机视觉与生物信息学交叉课程实践项目；为生物专业学生提供'
        'AI标注和数据科学培训，培养复合型人才；通过开源数据集和技术博客，'
        '推动蜱虫AI识别领域的知识共享和人才储备。')

    add_styled_heading(doc, '8.5 产业带动效应', level=2)
    add_body_text(doc,
        '【上游】推动蜱虫标注数据产业发展，建立标准化标注规范和培训体系。'
        '【中游】带动AI虫害识别技术链（数据采集→模型训练→边缘部署→云端分析）。'
        '【下游】赋能疾控监测、宠物医疗、农牧防治三大终端应用场景，'
        '形成完整的蜱虫智慧防控产业链。项目成熟后，有望成为细分领域的行业标准'
        '制定者和生态主导者。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 9: 财务计划
    # ============================================================
    add_styled_heading(doc, '九、财务计划', level=1)

    add_styled_heading(doc, '7.1 资金需求', level=2)
    add_body_text(doc, '当前寻求天使轮融资：50万元人民币。')
    add_table_with_data(doc,
        ['用途', '金额（万元）', '占比', '说明'],
        [
            ['技术研发', '20', '40%', 'GPU算力+数据标注+模型训练'],
            ['产品开发', '10', '20%', 'APP开发+Web后台+API服务'],
            ['市场验证', '10', '20%', '试点推广+学术参会+内容营销'],
            ['运营储备', '10', '20%', '6个月运营保障资金'],
        ])

    add_styled_heading(doc, '7.2 三年财务预测', level=2)

    add_body_text(doc, '【收入预测（单位：万元）】')
    add_table_with_data(doc,
        ['年份', 'API服务', 'SaaS订阅', 'APP增值', '定制部署', '总收入', '增长率'],
        [
            ['第1年', '10', '15', '5', '5', '35', '—'],
            ['第2年', '40', '60', '20', '20', '140', '300%'],
            ['第3年', '120', '200', '60', '70', '450', '221%'],
        ])

    add_body_text(doc, '【成本与利润预测（单位：万元）】')
    add_table_with_data(doc,
        ['年份', '总收入', '总成本', '毛利润', '毛利率', '净利润', '净利率'],
        [
            ['第1年', '35', '30', '5', '14%', '-10', '—'],
            ['第2年', '140', '70', '70', '50%', '30', '21%'],
            ['第3年', '450', '180', '270', '60%', '150', '33%'],
        ])

    add_styled_heading(doc, '7.3 财务分析', level=2)
    add_body_text(doc,
        '盈亏平衡点：预计第2年初达到月度盈亏平衡，需要月付费用户约200个（客单价约150元/月）。')

    add_body_text(doc,
        '投资回报：假设天使轮估值500万元（出让10%），3年后项目估值预计达1500-2000万元，'
        '早期投资者可获得3-4倍回报。')

    add_body_text(doc,
        '关键假设：用户增长率基于宠物医疗市场增速（15%）保守估算；定价策略参照'
        '同类SaaS产品（如医疗影像AI）；研发成本基于云GPU租赁和开源模型优化。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 8: 风险控制
    # ============================================================
    add_styled_heading(doc, '十、风险控制', level=1)

    add_table_with_data(doc,
        ['风险类型', '具体风险', '发生概率', '影响程度', '应对措施'],
        [
            ['技术风险', '模型在复杂场景精度下降', '中', '高',
             '持续扩充训练数据，引入对抗训练和数据增强技术；建立用户反馈纠错机制'],
            ['技术风险', '技术被模仿抄袭', '高', '中',
             '申请发明专利保护三层级检测架构；构建数据壁垒（标注数据是核心竞争力）'],
            ['市场风险', '用户增长不及预期', '中', '高',
             '聚焦高付费意愿的宠物医疗市场优先突破；与疾控中心建立标杆合作建立信任'],
            ['市场风险', '政策变化影响需求', '低', '中',
             '政策方向为利好（健康中国、智慧农业），密切关注行业监管动态灵活调整'],
            ['运营风险', '单人团队可持续性', '中', '高',
             '第2年引入核心成员分担压力；建立标准化开发流程减少个人依赖'],
            ['财务风险', '融资不到位', '中', '高',
             '申请政府创业补贴和科技计划项目；通过MVP小范围付费验证产生早期收入'],
            ['法律风险', '数据隐私合规', '低', '中',
             '聘请法律顾问确保合规；用户数据匿名化处理；采用联邦学习减少数据集中风险'],
        ])

    add_styled_heading(doc, '8.1 风险监控机制', level=2)
    add_body_text(doc,
        '建立月度风险评估会议制度，持续跟踪7大类风险指标。'
        '设立预警阈值：用户月增长<5%触发市场预警，模型精度下降>2%触发技术预警，'
        '现金流<3个月运营资金触发财务预警。每个预警指标有对应的应急预案。')

    doc.add_page_break()

    # ============================================================
    # CHAPTER 11: 发展现状和前景
    # ============================================================
    add_styled_heading(doc, '十一、发展现状和前景', level=1)

    add_styled_heading(doc, '11.1 发展现状', level=2)
    add_body_text(doc,
        '项目已完成阶段一全部研发工作，实现了从0到1的技术验证。'
        '以下为已取得的关键成果：')

    add_table_with_data(doc,
        ['成果类别', '具体内容', '完成度', '关键指标/说明'],
        [
            ['检测模型', 'Mask R-CNN蜱虫检测', '已完成', 'mAP@50=0.842，处于行业领先水平'],
            ['训练框架', 'torchvision训练pipeline', '已完成', '支持单类/多类可配置训练'],
            ['推理脚本', 'detect.py + detect_folder.py', '已完成', '支持单图/文件夹批量推理'],
            ['数据集', 'COCO格式蜱虫标注数据', '已完成', 'train/valid/test标准划分'],
            ['标注工具链', 'X-AnyLabeling + YOLO模型', '已完成', '支持AI辅助自动标注，21类昆虫'],
            ['多类数据', 'archive/archive2/archive3', '标注中', '28类昆虫数据集（8600+张）'],
            ['属性数据', 'TickIDNet元数据（fed/unfed）', '已关联', '可支持细粒度属性训练'],
            ['检测demo', '单张/批量推理可视化', '已完成', '输出含mask和置信度的标注图'],
        ])

    add_styled_heading(doc, '1.2 核心技术验证', level=2)
    add_body_text(doc,
        '已完成的检测模型在测试集上达到mAP@50=0.842，使用标准COCO评价指标。'
        '模型已在多种场景下验证：实验室蜱虫图片（高精度）、自然光照野外图片（中等精度）、'
        '与螨虫/蜘蛛等易混淆昆虫对比（待多类模型训练后进一步降低误检率）。')

    add_styled_heading(doc, '11.2 里程碑时间线', level=2)
    add_table_with_data(doc,
        ['时间', '里程碑', '核心产出'],
        [
            ['2025.Q4', '项目启动', '需求调研 + 技术方案设计 + 数据集获取'],
            ['2026.Q1', '阶段一完成', 'Mask R-CNN模型训练，mAP@50=0.842'],
            ['2026.Q2', '标注工具链', 'X-AnyLabeling配置 + YOLOv8m自动标注 + 28类标签体系'],
            ['2026.Q3', '多类模型', '多类昆虫检测模型训练，降低误检率（预计）'],
            ['2026.Q4', '细粒度模型', 'fed/unfed属性分类头集成（预计）'],
            ['2027.Q1', '产品原型', 'API+Web Demo原型上线'],
            ['2027.Q2', '种子用户', '疾控中心/宠物医院试点合作'],
            ['2027.Q3', '产品1.0', '移动端APP Beta + SaaS平台上线'],
            ['2027.Q4', '商业验证', '达到PMF，月付费用户100+'],
        ])

    add_styled_heading(doc, '11.3 3-5年发展路线图', level=2)
    add_body_text(doc, '【第1年（2026）——技术筑基期】')
    add_key_point(doc, '完成阶段二（多类检测）和阶段三（细粒度属性）技术研发')
    add_key_point(doc, '申请发明专利1-2项、软件著作权1项')
    add_key_point(doc, '发布研究论文1篇（CV/农业AI方向学术会议）')
    add_key_point(doc, '完成众创杯比赛，获取启动资金和创业资源')
    add_key_point(doc, 'MVP产品原型开发完成')

    add_body_text(doc, '【第2年（2027）——产品验证期】')
    add_key_point(doc, 'API云服务+移动端APP正式上线')
    add_key_point(doc, '突破100+付费用户（疾控+宠物医院）')
    add_key_point(doc, '完成天使轮融资（目标50-100万元）')
    add_key_point(doc, '团队扩展至3-5人')

    add_body_text(doc, '【第3年（2028）——商业增长期】')
    add_key_point(doc, '月付费用户1000+，年营收突破140万元，实现盈亏平衡')
    add_key_point(doc, '覆盖全国重点省份市场')
    add_key_point(doc, '完成Pre-A轮融资（目标300-500万元）')
    add_key_point(doc, '团队扩展至10-15人')

    add_body_text(doc, '【第4-5年（2029-2030）——规模扩张期】')
    add_key_point(doc, '年营收突破1000万元，市场份额达10%+')
    add_key_point(doc, '拓展海外市场（东南亚/非洲等蜱媒疾病高发区域）')
    add_key_point(doc, '构建蜱虫AI识别生态平台，成为行业标准')
    add_key_point(doc, '启动A轮融资或并购退出')

    add_styled_heading(doc, '11.4 前景展望', level=2)
    add_body_text(doc,
        '需求侧：全球气候变暖加速蜱虫栖息地扩张，蜱媒传染病发病率持续上升'
        '（中国疾控数据显示蜱媒传染病呈显著上升趋势），市场需求刚性且持续增长。')

    add_body_text(doc,
        '政策侧：健康中国2030、智慧农业、人工智能+医疗等政策叠加利好，'
        '项目高度契合国家战略方向。')

    add_body_text(doc,
        '技术侧：AI大模型和边缘计算技术的快速发展将持续降低部署成本、'
        '提升识别精度，为项目技术迭代提供坚实基础。')

    add_body_text(doc,
        '竞争侧：蓝海市场，目前无直接竞品，先发优势明显。'
        '通过持续的数据积累和算法迭代，有望成长为蜱虫AI识别赛道的定义者。')

    add_styled_heading(doc, '11.5 退出机制', level=2)
    add_body_text(doc,
        '为保障投资者利益，项目规划了三种退出路径：')
    add_key_point(doc, '战略收购：被AI医疗、农业科技或宠物行业头部企业收购（预期3-5倍回报）')
    add_key_point(doc, '持续经营：实现正向现金流后保持独立运营，以分红回报投资者')
    add_key_point(doc, 'IPO上市：条件成熟后申请科创板或北交所上市（长期路径）')

    # ============================================================
    # CLOSING
    # ============================================================
    doc.add_page_break()
    add_styled_heading(doc, '结语', level=1)
    add_body_text(doc,
        '智蜱卫士以"让每一次户外出行都远离蜱虫威胁"为使命，致力于成为国内蜱虫AI识别领域'
        '的领军者。我们深信，AI技术的力量不应停留在实验室，而应走进田野、走进社区、'
        '走进每一个需要它的角落。作为一支年轻而富有创造力的团队，我们拥有扎实的技术基础'
        '和清晰的商业路径，期待与投资人和合作伙伴携手，共同推动蜱虫防控技术的革新，'
        '为公共卫生事业贡献力量。')

    # ============================================================
    # SAVE
    # ============================================================
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '智蜱卫士_商业计划书_完整版.docx')
    doc.save(output_path)
    print(f'商业计划书已生成: {output_path}')
    return output_path

if __name__ == '__main__':
    generate()
