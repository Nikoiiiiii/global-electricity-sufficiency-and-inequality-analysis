from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SOURCE = Path('/Users/nico/Documents/PhD research/energy poverty/RQ4/Figure 4.png')
OUTPUT = Path('rq4_country_fe_outputs/Figure_4_heatmap_revised.png')

im = Image.open(SOURCE).convert('RGB')
sx, sy = im.width / 2048, im.height / 778

draw = ImageDraw.Draw(im)
font_path = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
font = ImageFont.truetype(font_path, round(15 * sy))

# Replace only the central coefficient text. The q-value stars in the upper-right
# corner of each cell and the original saturated negative colour are preserved.
replacements = [
    # (cover box, background sample point, text centre)
    ((1162, 578, 1274, 609), (1170, 603), (1218, 593)),  # gap: -626.48
    ((1760, 578, 1885, 609), (1770, 603), (1822, 593)),  # deficit: -533.23
]

for box, sample, centre in replacements:
    box = tuple(round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box))
    sample = (round(sample[0] * sx), round(sample[1] * sy))
    centre = (round(centre[0] * sx), round(centre[1] * sy))
    background = im.getpixel(sample)
    draw.rectangle(box, fill=background)
    label = 'OS'
    bounds = draw.textbbox((0, 0), label, font=font)
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text((centre[0] - tw / 2, centre[1] - th / 2 - 1), label,
              font=font, fill='white')

OUTPUT.parent.mkdir(exist_ok=True)
im.save(OUTPUT, dpi=(300, 300), optimize=True)
print(OUTPUT.resolve())
