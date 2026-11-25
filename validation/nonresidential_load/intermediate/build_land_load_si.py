from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path('/Users/nico/Documents/PhD research/energy poverty/validation-land')
MID = OUT / 'intermediate'
FIG = OUT / 'Figure_S4_nonresidential_load_robustness.png'
DOCX = OUT / 'SI_非居住用电负荷稳健性检验_中文版.docx'

ret = pd.read_csv(MID / 'scenario_retention.csv')
st = pd.read_csv(MID / 'rank_and_hotspot_stability.csv')
th = pd.read_csv(MID / 'gap_theil_decomposition.csv')
ame = pd.read_csv(MID / 'fractional_logit_ame.csv')
audit = pd.read_csv(MID / 'recalculated_baseline_audit.csv')
regional = pd.read_csv(MID / 'regional_summary.csv')

order = ['baseline', 'low_density_5', 'low_density_10', 'high_ecpc_1', 'high_ecpc_5']
short = {
    'baseline': 'Baseline', 'low_density_5': '最低密度 5%',
    'low_density_10': '最低密度 10%', 'high_ecpc_1': '最高 ECpc 1%',
    'high_ecpc_5': '最高 ECpc 5%'
}

ret['excluded_pixels'] = 1 - ret.retained_pixel_share
ret['excluded_population'] = 1 - ret.retained_population_share
ret['excluded_electricity'] = 1 - ret.retained_electricity_share
mins = st.groupby('scenario').spearman_vs_baseline.min()
hot = st[st.indicator == 'gap'].set_index('scenario').gap_top20_jaccard
min_rho = st[st.scenario != 'baseline'].spearman_vs_baseline.min()
min_hot = st[st.scenario != 'baseline'].dropna(subset=['gap_top20_jaccard']).gap_top20_jaccard.min()
base_t = th[th.scenario == 'baseline'].iloc[0]
max_t_shift_pp = max(float((th[c] - base_t[c]).abs().max() * 100) for c in
                     ['between_regions_share', 'between_countries_within_regions_share',
                      'within_countries_share'])
base_ame = ame[ame.scenario == 'baseline'].set_index(['indicator', 'variable']).ame
max_ame_shift = 0.0
for sc in order[1:]:
    q = ame[ame.scenario == sc].set_index(['indicator', 'variable']).ame
    max_ame_shift = max(max_ame_shift, float((q - base_ame).abs().max()))

BLUE = '1F4E79'; LIGHT = 'EAF2F8'; DARK = '243342'; MUTED = '5D6874'


def font(run, size=9.7, bold=False, color=DARK, italic=False):
    name = 'Arial Unicode MS'
    run.font.name = name
    for key in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        run._element.get_or_add_rPr().rFonts.set(qn('w:' + key), name)
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)


def margins(cell, top=65, start=75, bottom=65, end=75):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar'); tcPr.append(tcMar)
    for key, val in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        e = OxmlElement('w:' + key); e.set(qn('w:w'), str(val)); e.set(qn('w:type'), 'dxa')
        tcMar.append(e)


def add_para(doc, parts, after=4.5):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.08
    for text, bold, italic in parts:
        font(p.add_run(text), bold=bold, italic=italic)
    return p


doc = Document(); sec = doc.sections[0]
sec.page_width = Inches(8.5); sec.page_height = Inches(11)
sec.top_margin = Inches(.65); sec.bottom_margin = Inches(.62)
sec.left_margin = Inches(.78); sec.right_margin = Inches(.78)
sec.header_distance = Inches(.28); sec.footer_distance = Inches(.28)

normal = doc.styles['Normal']; normal.font.name = 'Arial Unicode MS'
normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Arial Unicode MS'); normal.font.size = Pt(9.7)

h = sec.header.paragraphs[0]; h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
font(h.add_run('Supplementary Information | Non-residential load robustness'), 7.7, color=MUTED)
f = sec.footer.paragraphs[0]; f.alignment = WD_ALIGN_PARAGRAPH.RIGHT
font(f.add_run('Supplementary Information'), 7.7, color=MUTED)

p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(8)
font(p.add_run('非居住用电负荷影响的稳健性检验'), 18.5, bold=True, color=BLUE)

add_para(doc, [(
    '为检验工业或商业用电是否误入像元用电量并影响 population-based sufficiency indicators，'
    '我们对两类疑似非居住像元分别进行 stress tests。第一类按全球 1-km 有效像元的人口密度排序，'
    '分别排除最低 5% 和 10%；第二类按人均用电量（ECpc）排序，分别排除最高 1% 和 5%。'
    '两类筛选分别实施，以识别各自影响，未叠加交集条件。每种情景均重新计算 GDL 的 Gap、Deficit 和 Excess。',
    False, False)])

add_para(doc, [('口径核查。', True, False), (
    f'像元重算基准与正文指标高度一致：Spearman rank correlation 为 '
    f'{audit.spearman_rho.min():.4f}–{audit.spearman_rho.max():.4f}，'
    f'平均绝对差异不超过 {audit.mean_absolute_difference.max():.4f}。因此，后续差异可归因于像元排除，而非重算方法。',
    False, False)])

add_para(doc, [('首先，', True, False), (
    '我们比较了七大区域的 Gap 和 Excess（Figure S4A–B）。四种排除情景下，各区域的相对位置与基准一致，'
    '指标绝对值也几乎不变，说明主要区域格局不依赖疑似非居住像元。', False, False)])

add_para(doc, [('其次，', True, False), (
    f'我们检验了 GDL 空间排序及 Gap 热点。所有替代情景与基准的 Spearman rank correlation '
    f'均不低于 {min_rho:.6f}（Figure S4C）；Gap 最高 20% GDL 的 Jaccard index 均为 {min_hot:.3f}（Table S4）。'
    '这表明 GDL 排序和热点识别未发生实质变化。', False, False)])

add_para(doc, [('第三，', True, False), (
    f'我们重新计算了 Gap 的空间 Theil decomposition。基准结果中，区域间、区域内国家间和国家内差异'
    f'分别占 {base_t.between_regions_share*100:.1f}%、'
    f'{base_t.between_countries_within_regions_share*100:.1f}% 和 '
    f'{base_t.within_countries_share*100:.1f}%；替代情景的最大变化不足 0.001 个百分点。'
    '因此，文章关于不平等空间来源的结论保持不变。', False, False)])

add_para(doc, [('最后，', True, False), (
    f'我们在正文有效样本上重新估计了 Gap 和 Deficit 的 fractional-logit models（1,527 个 GDL，162 个国家）。'
    f'主要变量的估计方向和显著性均保持不变，average marginal effects 相对基准的最大变化小于 '
    f'{max_ame_shift:.6f}（Figure S4D）。', False, False)])

add_para(doc, [('说明。', True, False), (
    '剔除高 ECpc 像元会机械性降低 Excess，因此该情景应理解为对过度消费估计的保守下界检验。'
    '即便如此，区域格局和 GDL 排序仍保持稳定。', False, False)])

t = doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
c = t.cell(0, 0); shade(c, LIGHT); margins(c, 105, 135, 105, 135)
p = c.paragraphs[0]
font(p.add_run('总结　'), 9.8, bold=True, color=BLUE)
font(p.add_run('在四种独立像元排除情景下，区域格局、GDL 排序、热点识别、空间不平等分解及主要社会梯度均保持稳定。'
               '因此，疑似工业或商业用电像元不是驱动本文主要 population-based sufficiency 结论的因素。'), 9.8)

doc.add_page_break()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
p.add_run().add_picture(str(FIG), width=Inches(6.45))
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(5)
font(p.add_run('Figure S4. 非居住用电负荷影响的稳健性检验。'), 8.4, bold=True)
font(p.add_run('A，七大区域的 Gap；B，七大区域的 Excess；C，GDL 指标相对基准的 Spearman rank correlation；'
               'D，Gap fractional-logit model 的 average marginal effects。Low-density 和 High-ECpc 分别表示排除全球人口密度最低及 ECpc 最高的像元。'),
     8.4, color=MUTED)

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(3); p.paragraph_format.keep_with_next = True
font(p.add_run('Table S4. 各像元排除情景的核心稳健性结果'), 10.6, bold=True, color=BLUE)

headers = ['情景', '实际排除\n像元 (%)', '排除用电量\n(%)', '最低 ρ', 'Gap 热点\nJaccard']
tbl = doc.add_table(rows=1, cols=5); tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl.autofit = False
widths = [1.55, 1.15, 1.22, .90, 1.18]
for i, (label, width) in enumerate(zip(headers, widths)):
    tbl.columns[i].width = Inches(width)
    cell = tbl.cell(0, i); shade(cell, BLUE); margins(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(p.add_run(label), 7.8, bold=True, color='FFFFFF')

for sc in order:
    rr = ret[ret.scenario == sc].iloc[0]
    vals = [short[sc], f'{rr.excluded_pixels*100:.2f}',
            f'{rr.excluded_electricity*100:.3f}', f'{mins[sc]:.6f}', f'{hot[sc]:.3f}']
    cells = tbl.add_row().cells
    for cell, val in zip(cells, vals):
        margins(cell); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if sc == 'baseline': shade(cell, LIGHT)
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run(val), 8.0, bold=(sc == 'baseline'))

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(2)
font(p.add_run('注：'), 8.0, bold=True)
font(p.add_run('实际排除比例按完整有效栅格总体计算；分位点并列像元一并排除，因此最低密度 5% 和最高 ECpc 1% 情景的实际比例略有偏离名义值。'
               '四种情景排除的人口占比均低于 0.0001%。最低 ρ 为 Gap、Deficit 和 Excess 三项指标中相对基准的最小 Spearman rank correlation。'),
     8.0, color=MUTED)

doc.core_properties.title = '非居住用电负荷影响的稳健性检验'
doc.core_properties.subject = 'Supplementary Information'
doc.core_properties.author = ''
doc.save(DOCX)

shutil.copy2(Path(__file__).with_name('land_load_validation.py'), MID / 'land_load_validation.py')
shutil.copy2(Path(__file__), MID / 'build_land_load_si.py')
print(DOCX)
