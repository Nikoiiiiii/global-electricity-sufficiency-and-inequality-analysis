from pathlib import Path
import json
import os
import re
import sys
import time
import warnings

PROJECT = Path('/Users/nico/Research/Energy sufficiency')
LOCAL_PKG = PROJECT / 'Process/python_packages'
sys.path.insert(0, str(LOCAL_PKG))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mpl-land-load-validation')

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from scipy.stats import spearmanr
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

OUT = Path('/Users/nico/Documents/PhD research/energy poverty/validation-land')
MID = OUT / 'intermediate'
PIX = MID / 'pixel_cache'
EC_PATH = Path('/Users/nico/Documents/PhD research/energy poverty/Data/EC2015.tif')
POP_PATH = Path('/Users/nico/Documents/PhD research/energy poverty/Data/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0.tif')
SHP_PATH = Path('/Users/nico/Documents/PhD research/energy poverty/Data/GDL Shapefiles V4/GDL Shapefiles V4.shp')
DATA_PATH = PROJECT / 'Data/global_SE_EC_sufficiency_controls_2015.csv'

LOWER = 300.0
UPPER = 5000.0
SCENARIOS = [
    ('baseline', 'Baseline'),
    ('low_density_5', 'Exclude lowest 5% density'),
    ('low_density_10', 'Exclude lowest 10% density'),
    ('high_ecpc_1', 'Exclude highest 1% ECpc'),
    ('high_ecpc_5', 'Exclude highest 5% ECpc'),
]
REGION_ORDER = [
    'Sub-Saharan Africa', 'South Asia', 'Eastern and South-eastern Asia',
    'Northern Africa and Western Asia', 'Latin America and the Caribbean',
    'Europe', 'Northern America and Oceania'
]
REGION_SHORT = {
    'Sub-Saharan Africa': 'SSA', 'South Asia': 'South Asia',
    'Eastern and South-eastern Asia': 'E & SE Asia',
    'Northern Africa and Western Asia': 'N. Africa & W. Asia',
    'Latin America and the Caribbean': 'Latin America & Caribbean',
    'Europe': 'Europe', 'Northern America and Oceania': 'N. America & Oceania'
}
REGION_COLORS = {
    'Sub-Saharan Africa': '#B2182B', 'South Asia': '#EF8A62',
    'Eastern and South-eastern Asia': '#67A9CF',
    'Northern Africa and Western Asia': '#D9A441',
    'Latin America and the Caribbean': '#5AB4AC',
    'Europe': '#8073AC', 'Northern America and Oceania': '#2166AC'
}


def clipped_window(src, geom):
    w = from_bounds(*geom.bounds, transform=src.transform)
    c0 = max(0, int(np.floor(w.col_off)))
    r0 = max(0, int(np.floor(w.row_off)))
    c1 = min(src.width, int(np.ceil(w.col_off + w.width)))
    r1 = min(src.height, int(np.ceil(w.row_off + w.height)))
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def cache_name(code):
    return PIX / (re.sub(r'[^A-Za-z0-9_.-]+', '_', str(code)) + '.npz')


def build_pixel_cache():
    """Create one restart-safe file per GDL using a single aggregation method."""
    PIX.mkdir(parents=True, exist_ok=True)
    shp = gpd.read_file(SHP_PATH)[['GDLcode', 'geometry']]
    rows = []
    with rasterio.open(EC_PATH) as ec, rasterio.open(POP_PATH) as pop:
        shp = shp.to_crs(ec.crs)

        def population_for_tile(tile):
            left, bottom, right, top = rasterio.windows.bounds(tile, ec.transform)
            sw = from_bounds(left, bottom, right, top, pop.transform)
            c0 = max(0, int(np.floor(sw.col_off)) - 2)
            r0 = max(0, int(np.floor(sw.row_off)) - 2)
            c1 = min(pop.width, int(np.ceil(sw.col_off + sw.width)) + 2)
            r1 = min(pop.height, int(np.ceil(sw.row_off + sw.height)) + 2)
            srcwin = Window(c0, r0, c1 - c0, r1 - r0)
            src = pop.read(1, window=srcwin, masked=False).astype('float64')
            src[np.isclose(src, pop.nodata) | ~np.isfinite(src) | (src < 0)] = 0.0
            st = pop.window_transform(srcwin)
            dt = ec.window_transform(tile)
            xs = st.c + (np.arange(src.shape[1]) + 0.5) * st.a
            ys = st.f + (np.arange(src.shape[0]) + 0.5) * st.e
            dc = np.floor((xs - dt.c) / dt.a).astype(int)
            dr = np.floor((dt.f - ys) / (-dt.e)).astype(int)
            rr, cc = np.meshgrid(dr, dc, indexing='ij')
            good = (rr >= 0) & (rr < int(tile.height)) & (cc >= 0) & (cc < int(tile.width))
            flat = rr[good] * int(tile.width) + cc[good]
            dst = np.bincount(flat, weights=src[good],
                              minlength=int(tile.height * tile.width)).reshape(
                                  int(tile.height), int(tile.width)).astype('float64')
            dst[dst <= 0] = np.nan
            return dst

        started = time.time()
        for i, rec in shp.iterrows():
            code = str(rec.GDLcode)
            target = cache_name(code)
            if target.exists():
                try:
                    with np.load(target) as q:
                        rows.append({'GDLcode': code, 'n_pixels': len(q['population']),
                                     'cache': target.name, 'status': 'ok'})
                    continue
                except Exception:
                    target.unlink(missing_ok=True)
            geom = rec.geometry
            if geom is None or geom.is_empty:
                rows.append({'GDLcode': code, 'n_pixels': 0, 'cache': target.name,
                             'status': 'empty_geometry'})
                continue
            win = clipped_window(ec, geom)
            if win is None:
                rows.append({'GDLcode': code, 'n_pixels': 0, 'cache': target.name,
                             'status': 'no_window'})
                continue
            geom_mapping = geom.__geo_interface__
            pop_parts, ecpc_parts = [], []
            block = 384
            for roff in range(int(win.row_off), int(win.row_off + win.height), block):
                h = min(block, int(win.row_off + win.height) - roff)
                for coff in range(int(win.col_off), int(win.col_off + win.width), block):
                    w = min(block, int(win.col_off + win.width) - coff)
                    tile = Window(coff, roff, w, h)
                    e = ec.read(1, window=tile, masked=True).astype('float64')
                    p = population_for_tile(tile)
                    inside = geometry_mask([geom_mapping], out_shape=e.shape,
                                           transform=ec.window_transform(tile),
                                           invert=True, all_touched=False)
                    ev = np.asarray(e.filled(np.nan))
                    valid = inside & np.isfinite(ev) & np.isfinite(p) & (p > 0) & (ev >= 0)
                    if valid.any():
                        pv = p[valid].astype('float32')
                        pop_parts.append(pv)
                        ecpc_parts.append((ev[valid] / pv).astype('float32'))
            if pop_parts:
                p_all = np.concatenate(pop_parts)
                x_all = np.concatenate(ecpc_parts)
            else:
                p_all = np.empty(0, dtype='float32')
                x_all = np.empty(0, dtype='float32')
            np.savez_compressed(target, population=p_all, ecpc=x_all)
            rows.append({'GDLcode': code, 'n_pixels': len(p_all), 'cache': target.name,
                         'status': 'ok' if len(p_all) else 'no_valid_pixels'})
            if (i + 1) % 25 == 0:
                pd.DataFrame(rows).to_csv(MID / 'pixel_cache_manifest_checkpoint.csv', index=False)
                mins = (time.time() - started) / 60
                print(f'cache {i + 1}/{len(shp)}; elapsed {mins:.1f} min', flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MID / 'pixel_cache_manifest.csv', index=False)
    return manifest


def create_global_arrays(manifest):
    good = manifest[(manifest.status == 'ok') & (manifest.n_pixels > 0)].copy()
    total_n = int(good.n_pixels.sum())
    p_path = MID / 'global_population_float32.npy'
    x_path = MID / 'global_ecpc_float32.npy'
    meta_path = MID / 'global_arrays_meta.json'
    rebuild = True
    if p_path.exists() and x_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        rebuild = meta.get('n_pixels') != total_n
    if rebuild:
        p_mm = np.lib.format.open_memmap(p_path, mode='w+', dtype='float32', shape=(total_n,))
        x_mm = np.lib.format.open_memmap(x_path, mode='w+', dtype='float32', shape=(total_n,))
        cursor = 0
        for j, rec in good.reset_index(drop=True).iterrows():
            with np.load(PIX / rec.cache) as q:
                n = len(q['population'])
                p_mm[cursor:cursor+n] = q['population']
                x_mm[cursor:cursor+n] = q['ecpc']
                cursor += n
            if (j + 1) % 200 == 0:
                print(f'global arrays {j + 1}/{len(good)}', flush=True)
        p_mm.flush(); x_mm.flush()
        meta_path.write_text(json.dumps({'n_pixels': total_n}, indent=2))
        del p_mm, x_mm
    return p_path, x_path, total_n


def exact_cutoffs(p_path, x_path, n_pixels):
    cutoff_path = MID / 'global_pixel_cutoffs.json'
    if cutoff_path.exists():
        prior = json.loads(cutoff_path.read_text())
        if prior.get('n_pixels') == n_pixels:
            return prior
    p = np.load(p_path, mmap_mode='r')
    pvals = np.quantile(np.asarray(p), [0.05, 0.10], method='linear')
    del p
    x = np.load(x_path, mmap_mode='r')
    xvals = np.quantile(np.asarray(x), [0.95, 0.99], method='linear')
    del x
    out = {'n_pixels': n_pixels, 'population_q05': float(pvals[0]),
           'population_q10': float(pvals[1]), 'ecpc_q95': float(xvals[0]),
           'ecpc_q99': float(xvals[1]),
           'definition': 'Global quantiles among valid 1-km pixels with population > 0.'}
    cutoff_path.write_text(json.dumps(out, indent=2))
    return out


def indicators(p, x, keep):
    w = p[keep].astype('float64')
    z = x[keep].astype('float64')
    denom = w.sum()
    if denom <= 0:
        return {k: np.nan for k in ['gap', 'deficit', 'excess', 'retained_population',
                                     'retained_electricity'] } | {'retained_pixels': 0}
    return {
        'gap': float(np.sum(w * (z < LOWER)) / denom),
        'deficit': float(np.sum(w * np.maximum(0.0, 1.0 - z / LOWER)) / denom),
        'excess': float(np.sum(w * (z > UPPER)) / denom),
        'retained_population': float(denom),
        'retained_electricity': float(np.sum(w * z)),
        'retained_pixels': int(keep.sum())
    }


def compute_gdl_scenarios(manifest, cutoffs):
    rows = []
    for j, rec in manifest.iterrows():
        if rec.status != 'ok' or rec.n_pixels <= 0:
            continue
        with np.load(PIX / rec.cache) as q:
            p = q['population']
            x = q['ecpc']
        masks = {
            'baseline': np.ones(len(p), dtype=bool),
            'low_density_5': p > cutoffs['population_q05'],
            'low_density_10': p > cutoffs['population_q10'],
            # Exclude the quantile boundary as well. This is deterministic and
            # conservative when float32 values are tied at the cut-off.
            'high_ecpc_1': x < cutoffs['ecpc_q99'],
            'high_ecpc_5': x < cutoffs['ecpc_q95'],
        }
        for scenario, label in SCENARIOS:
            row = {'GDLcode': rec.GDLcode, 'scenario': scenario,
                   'scenario_label': label, 'total_pixels': len(p),
                   'total_population': float(p.astype('float64').sum()),
                   'total_electricity': float(np.sum(p.astype('float64') * x.astype('float64')))}
            row.update(indicators(p, x, masks[scenario]))
            rows.append(row)
        if (j + 1) % 250 == 0:
            print(f'indicators {j + 1}/{len(manifest)}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(MID / 'gdl_scenario_indicators_long.csv', index=False)
    wide = out.pivot(index='GDLcode', columns='scenario',
                     values=['gap', 'deficit', 'excess', 'retained_population',
                             'retained_electricity', 'retained_pixels'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide.reset_index().to_csv(MID / 'gdl_scenario_indicators_wide.csv', index=False)
    return out


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def prepare(long):
    base = pd.read_csv(DATA_PATH)
    d = long.merge(base, on='GDLcode', how='inner', validate='many_to_one')
    d['income_clean'] = d['income'].where(d['income'] >= 0, np.nan)
    d['wealth_clean'] = d['wealth'].where(d['wealth'] >= 0, np.nan)
    d['SES_raw'] = d['income_clean'].combine_first(d['wealth_clean'])
    d['female_share'] = d['gender']
    d['GDP_pc'] = d['GDP'] / d['population'].replace(0, np.nan)
    d['log_GDP_pc'] = np.log(d['GDP_pc'].where(d['GDP_pc'] > 0))
    d['log_population'] = np.log(d['population'].where(d['population'] > 0))
    d.to_csv(MID / 'analysis_long.csv', index=False)
    return d


def scenario_retention(d):
    rows = []
    for scenario, label in SCENARIOS:
        z = d[d.scenario == scenario]
        rows.append({
            'scenario': scenario, 'scenario_label': label,
            'retained_pixel_share': z.retained_pixels.sum() / z.total_pixels.sum(),
            'retained_population_share': z.retained_population.sum() / z.total_population.sum(),
            'retained_electricity_share': z.retained_electricity.sum() / z.total_electricity.sum(),
            'n_gdl': z.GDLcode.nunique()
        })
    out = pd.DataFrame(rows)
    out.to_csv(MID / 'scenario_retention.csv', index=False)
    return out


def regional_summary(d):
    rows = []
    for scenario, label in SCENARIOS:
        z0 = d[d.scenario == scenario]
        for region in ['Global'] + REGION_ORDER:
            z = z0 if region == 'Global' else z0[z0.analytical_region == region]
            zg = z[z.sufficiency_gap.notna()]
            zd = z[z.sufficiency_deficit.notna()]
            ze = z[z.sufficiency_excess.notna()]
            rows.append({'scenario': scenario, 'scenario_label': label, 'region': region,
                         'gap': weighted_mean(zg.gap, zg.retained_population),
                         'deficit': weighted_mean(zd.deficit, zd.retained_population),
                         'excess': weighted_mean(ze.excess, ze.retained_population),
                         'population': z.retained_population.sum(),
                         'n_gdl': z.GDLcode.nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(MID / 'regional_summary.csv', index=False)
    return out


def stability(d):
    published = {'gap': 'sufficiency_gap', 'deficit': 'sufficiency_deficit',
                 'excess': 'sufficiency_excess'}
    keep_cols = ['GDLcode', 'gap', 'deficit', 'excess'] + list(published.values())
    b = d[d.scenario == 'baseline'][keep_cols]
    rows = []
    for scenario, label in SCENARIOS:
        z = d[d.scenario == scenario][['GDLcode', 'gap', 'deficit', 'excess']]
        q = b.merge(z, on='GDLcode', suffixes=('_base', '_alt'), validate='one_to_one')
        for metric in ['gap', 'deficit', 'excess']:
            pub = published[metric]
            m = q[f'{metric}_base'].notna() & q[f'{metric}_alt'].notna() & q[pub].notna()
            rho = spearmanr(q.loc[m, f'{metric}_base'], q.loc[m, f'{metric}_alt']).statistic
            rows.append({'scenario': scenario, 'scenario_label': label,
                         'indicator': metric, 'spearman_vs_baseline': float(rho),
                         'n_gdl': int(m.sum())})
        m = q.gap_base.notna() & q.gap_alt.notna() & q.sufficiency_gap.notna()
        n_hot = max(1, int(np.ceil(m.sum() * 0.20)))
        hot_b = set(q.loc[m].nlargest(n_hot, 'gap_base').GDLcode)
        hot_a = set(q.loc[m].nlargest(n_hot, 'gap_alt').GDLcode)
        jac = len(hot_b & hot_a) / len(hot_b | hot_a)
        rows[-3]['gap_top20_jaccard'] = jac
        rows[-3]['gap_hotspots_common'] = len(hot_b & hot_a)
        rows[-3]['gap_hotspots_each'] = n_hot
    out = pd.DataFrame(rows)
    out.to_csv(MID / 'rank_and_hotspot_stability.csv', index=False)
    return out


def theil_t(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x >= 0)
    x = x[m]; w = w[m]
    if not len(x): return np.nan
    p = w / w.sum(); mu = np.sum(p * x)
    if mu <= 0: return 0.0
    r = x / mu
    return float(np.sum(p[r > 0] * r[r > 0] * np.log(r[r > 0])))


def group_decomposition(z, value, weight, group):
    z = z[[value, weight, group]].dropna()
    z = z[(z[weight] > 0) & (z[value] >= 0)]
    W = z[weight].sum(); mu = np.average(z[value], weights=z[weight])
    if W <= 0 or mu <= 0: return 0.0, 0.0
    between = 0.0; within = 0.0
    for _, g in z.groupby(group):
        Wg = g[weight].sum(); mug = np.average(g[value], weights=g[weight])
        scale = (Wg / W) * (mug / mu) if mug > 0 else 0.0
        if scale > 0: between += scale * np.log(mug / mu)
        within += scale * theil_t(g[value], g[weight])
    return float(between), float(within)


def spatial_decomposition(z):
    total = theil_t(z.gap, z.retained_population)
    br, wr = group_decomposition(z, 'gap', 'retained_population', 'analytical_region')
    W = z.retained_population.sum(); mu = np.average(z.gap, weights=z.retained_population)
    bc_emb = 0.0; wc_emb = 0.0
    for _, r in z.groupby('analytical_region'):
        Wr = r.retained_population.sum(); mur = np.average(r.gap, weights=r.retained_population)
        scale = (Wr / W) * (mur / mu) if mur > 0 and mu > 0 else 0.0
        bc, wc = group_decomposition(r, 'gap', 'retained_population', 'iso_code')
        bc_emb += scale * bc; wc_emb += scale * wc
    return {'theil_total': total, 'between_regions': br,
            'between_countries_within_regions': bc_emb,
            'within_countries': wc_emb,
            'component_sum': br + bc_emb + wc_emb}


def theil_results(d):
    rows = []
    for scenario, label in SCENARIOS:
        z = d[d.scenario == scenario].dropna(subset=['gap', 'sufficiency_gap',
                                                      'retained_population',
                                                      'analytical_region', 'iso_code'])
        row = {'scenario': scenario, 'scenario_label': label, 'n_gdl': len(z)}
        row.update(spatial_decomposition(z))
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ['between_regions', 'between_countries_within_regions', 'within_countries']:
        out[c + '_share'] = out[c] / out.theil_total
    out.to_csv(MID / 'gap_theil_decomposition.csv', index=False)
    return out


def fractional_models(d):
    predictors = ['SES_raw', 'rural', 'education', 'female_share', 'log_GDP_pc',
                  'log_population', 'total_HDD', 'total_CDD']
    wide = d.pivot(index='GDLcode', columns='scenario', values=['gap', 'deficit'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    cov = d[d.scenario == 'baseline'][['GDLcode', 'iso_code', 'sufficiency_gap',
                                       'sufficiency_deficit'] + predictors].drop_duplicates('GDLcode')
    q = cov.merge(wide.reset_index(), on='GDLcode', how='inner', validate='one_to_one')
    outcomes = [f'{m}_{s}' for m in ['gap', 'deficit'] for s, _ in SCENARIOS]
    sample = q.dropna(subset=predictors + ['iso_code', 'sufficiency_gap',
                                           'sufficiency_deficit'] + outcomes).copy()
    for v in predictors:
        sample['z_' + v] = (sample[v] - sample[v].mean()) / sample[v].std(ddof=0)
    rhs = ' + '.join('z_' + v for v in predictors) + ' + C(iso_code)'
    rows = []
    for metric in ['gap', 'deficit']:
        for scenario, label in SCENARIOS:
            outcome = f'{metric}_{scenario}'
            fit = smf.glm(outcome + ' ~ ' + rhs, data=sample,
                          family=sm.families.Binomial()).fit(
                              cov_type='cluster', cov_kwds={'groups': sample.iso_code},
                              maxiter=300)
            pred = fit.predict(sample)
            factor = float(np.mean(pred * (1 - pred)))
            ci = fit.conf_int()
            for v in predictors:
                p = 'z_' + v
                rows.append({'indicator': metric, 'scenario': scenario,
                             'scenario_label': label, 'variable': v,
                             'ame': float(fit.params[p] * factor),
                             'ame_low': float(ci.loc[p, 0] * factor),
                             'ame_high': float(ci.loc[p, 1] * factor),
                             'p_value': float(fit.pvalues[p]), 'n_gdl': len(sample),
                             'n_countries': sample.iso_code.nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(MID / 'fractional_logit_ame.csv', index=False)
    return out


def baseline_audit(d):
    z = d[d.scenario == 'baseline']
    rows = []
    for metric, published in [('gap', 'sufficiency_gap'), ('deficit', 'sufficiency_deficit'),
                              ('excess', 'sufficiency_excess')]:
        m = z[metric].notna() & z[published].notna()
        rows.append({'indicator': metric, 'n_gdl': int(m.sum()),
                     'spearman_rho': float(spearmanr(z.loc[m, metric], z.loc[m, published]).statistic),
                     'mean_absolute_difference': float(np.mean(np.abs(z.loc[m, metric] - z.loc[m, published])))})
    out = pd.DataFrame(rows)
    out.to_csv(MID / 'recalculated_baseline_audit.csv', index=False)
    return out


def make_core_table(retention, stability, theil):
    rank = stability.pivot(index='scenario', columns='indicator', values='spearman_vs_baseline')
    hot = stability[stability.indicator == 'gap'].set_index('scenario')['gap_top20_jaccard']
    t = retention.set_index('scenario').join(rank).join(hot).join(
        theil.set_index('scenario')[['theil_total', 'between_regions_share',
                                     'between_countries_within_regions_share',
                                     'within_countries_share']])
    t = t.reset_index()
    t.to_csv(OUT / 'Table_S4_nonresidential_load_robustness.csv', index=False)
    return t


def make_figure(regional, stability, ame):
    sns.set_theme(style='whitegrid', font='DejaVu Sans')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    x = np.arange(len(SCENARIOS))
    xlabels = ['Baseline', 'Low-density\n5%', 'Low-density\n10%',
               'High-ECpc\n1%', 'High-ECpc\n5%']
    for ax, metric, title, ylabel in [
        (axes[0, 0], 'gap', 'A  Regional insufficiency gap', 'Population share'),
        (axes[0, 1], 'excess', 'B  Regional excessive consumption', 'Population share')]:
        for region in REGION_ORDER:
            z = regional[regional.region == region].set_index('scenario').reindex([s for s, _ in SCENARIOS])
            ax.plot(x, z[metric], marker='o', ms=4, lw=1.7,
                    label=REGION_SHORT[region], color=REGION_COLORS[region])
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel(ylabel); ax.set_xticks(x, xlabels)
        ax.tick_params(axis='x', labelsize=8)
    axes[0, 0].legend(fontsize=7.3, ncol=2, frameon=False, loc='best')
    ax = axes[1, 0]
    for metric, colour in [('gap', '#B2182B'), ('deficit', '#D98C00'), ('excess', '#2166AC')]:
        z = stability[stability.indicator == metric].set_index('scenario').reindex([s for s, _ in SCENARIOS])
        ax.plot(x, z.spearman_vs_baseline, marker='o', lw=2, label=metric.capitalize(), color=colour)
    minrho = stability.spearman_vs_baseline.min()
    ax.set_ylim(max(0, minrho - 0.04), 1.005)
    ax.set_xticks(x, xlabels); ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('Spearman correlation vs baseline')
    ax.set_title('C  GDL rank stability', loc='left', fontweight='bold')
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax = axes[1, 1]
    labels = {'SES_raw': 'SES', 'rural': 'Rurality', 'education': 'Education',
              'female_share': 'Female share'}
    colours = {'SES_raw': '#5AB4AC', 'rural': '#2166AC',
               'education': '#8073AC', 'female_share': '#B2182B'}
    for v, label in labels.items():
        z = ame[(ame.indicator == 'gap') & (ame.variable == v)].set_index('scenario').reindex([s for s, _ in SCENARIOS])
        ax.plot(x, z.ame, marker='o', lw=2, label=label, color=colours[v])
    ax.axhline(0, color='0.35', lw=0.8)
    ax.set_xticks(x, xlabels); ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('Average marginal effect')
    ax.set_title('D  Fractional-logit estimates for Gap', loc='left', fontweight='bold')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(OUT / 'Figure_S4_nonresidential_load_robustness.png', dpi=350, bbox_inches='tight')
    fig.savefig(OUT / 'Figure_S4_nonresidential_load_robustness.pdf', bbox_inches='tight')
    plt.close(fig)


def write_summary(cutoffs, retention, stability, theil, audit, ame):
    summary = {
        'cutoffs': cutoffs,
        'retention': retention.to_dict('records'),
        'minimum_spearman_by_indicator': stability.groupby('indicator').spearman_vs_baseline.min().to_dict(),
        'minimum_gap_hotspot_jaccard': float(stability[stability.indicator == 'gap'].gap_top20_jaccard.min()),
        'theil': theil.to_dict('records'),
        'baseline_audit': audit.to_dict('records'),
        'regression_sample_n': int(ame.n_gdl.iloc[0]),
        'regression_countries_n': int(ame.n_countries.iloc[0])
    }
    (MID / 'validation_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True); MID.mkdir(parents=True, exist_ok=True)
    manifest = build_pixel_cache()
    p_path, x_path, n = create_global_arrays(manifest)
    cutoffs = exact_cutoffs(p_path, x_path, n)
    print('cutoffs', cutoffs, flush=True)
    long = compute_gdl_scenarios(manifest, cutoffs)
    d = prepare(long)
    # Retention statistics describe the complete raster universe used to set
    # the global quantiles; inferential checks below use the published sample.
    retention = scenario_retention(long)
    regional = regional_summary(d)
    stable = stability(d)
    theil = theil_results(d)
    ame = fractional_models(d)
    audit = baseline_audit(d)
    make_core_table(retention, stable, theil)
    make_figure(regional, stable, ame)
    write_summary(cutoffs, retention, stable, theil, audit, ame)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
