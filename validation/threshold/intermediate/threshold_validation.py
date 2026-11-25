from pathlib import Path
import json, os, sys, warnings

PROJECT = Path('/Users/nico/Research/Energy sufficiency')
LOCAL_PKG = PROJECT / 'Process/python_packages'
sys.path.insert(0, str(LOCAL_PKG))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mpl-threshold-validation')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

OUT = Path('/Users/nico/Documents/PhD research/energy poverty/Validation- threshold')
MID = OUT / 'intermediate'
OUT.mkdir(parents=True, exist_ok=True)
MID.mkdir(parents=True, exist_ok=True)

EC_PATH = Path('/Users/nico/Documents/PhD research/energy poverty/Data/EC2015.tif')
POP_PATH = Path('/Users/nico/Documents/PhD research/energy poverty/Data/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0.tif')
SHP_PATH = Path('/Users/nico/Documents/PhD research/energy poverty/Data/GDL Shapefiles V4/GDL Shapefiles V4.shp')
DATA_PATH = PROJECT / 'Data/global_SE_EC_sufficiency_controls_2015.csv'

LOWER = [200, 250, 300, 400, 500]
UPPER = [4000, 4500, 5000, 5500, 6000]
REGION_ORDER = [
    'Sub-Saharan Africa', 'South Asia', 'Eastern and South-eastern Asia',
    'Northern Africa and Western Asia', 'Latin America and the Caribbean',
    'Europe', 'Northern America and Oceania'
]
REGION_SHORT = {
    'Sub-Saharan Africa':'SSA', 'South Asia':'South Asia',
    'Eastern and South-eastern Asia':'E & SE Asia',
    'Northern Africa and Western Asia':'N. Africa & W. Asia',
    'Latin America and the Caribbean':'Latin America & Caribbean',
    'Europe':'Europe', 'Northern America and Oceania':'N. America & Oceania'
}
REGION_COLORS = {
    'Sub-Saharan Africa':'#B2182B', 'South Asia':'#EF8A62',
    'Eastern and South-eastern Asia':'#67A9CF',
    'Northern Africa and Western Asia':'#D9A441',
    'Latin America and the Caribbean':'#5AB4AC',
    'Europe':'#8073AC', 'Northern America and Oceania':'#2166AC'
}


def clipped_window(src, geom):
    b = geom.bounds
    w = from_bounds(*b, transform=src.transform)
    c0 = max(0, int(np.floor(w.col_off)))
    r0 = max(0, int(np.floor(w.row_off)))
    c1 = min(src.width, int(np.ceil(w.col_off + w.width)))
    r1 = min(src.height, int(np.ceil(w.row_off + w.height)))
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1-c0, r1-r0)


def compute_pixel_thresholds():
    shp = gpd.read_file(SHP_PATH)[['GDLcode','geometry']]
    rows=[]
    with rasterio.open(EC_PATH) as ec, rasterio.open(POP_PATH) as pop:
        shp = shp.to_crs(ec.crs)
        def population_for_tile(tile):
            left,bottom,right,top=rasterio.windows.bounds(tile,ec.transform)
            sw=from_bounds(left,bottom,right,top,pop.transform)
            c0=max(0,int(np.floor(sw.col_off))-2); r0=max(0,int(np.floor(sw.row_off))-2)
            c1=min(pop.width,int(np.ceil(sw.col_off+sw.width))+2); r1=min(pop.height,int(np.ceil(sw.row_off+sw.height))+2)
            srcwin=Window(c0,r0,c1-c0,r1-r0)
            src=pop.read(1,window=srcwin,masked=False).astype('float64')
            # Population is extensive. Assign each 250 m source-cell centre to its
            # containing 1 km EC cell and sum, avoiding interpolation and GDAL's
            # very slow SUM warp path for a few source blocks.
            src[np.isclose(src,pop.nodata) | ~np.isfinite(src) | (src<0)] = 0.0
            st=pop.window_transform(srcwin); dt=ec.window_transform(tile)
            xs=st.c+(np.arange(src.shape[1])+.5)*st.a
            ys=st.f+(np.arange(src.shape[0])+.5)*st.e
            dc=np.floor((xs-dt.c)/dt.a).astype(int)
            dr=np.floor((dt.f-ys)/(-dt.e)).astype(int)
            rr,cc=np.meshgrid(dr,dc,indexing='ij')
            good=(rr>=0)&(rr<int(tile.height))&(cc>=0)&(cc<int(tile.width))
            flat=rr[good]*int(tile.width)+cc[good]
            dst=np.bincount(flat,weights=src[good],minlength=int(tile.height*tile.width)).reshape(int(tile.height),int(tile.width)).astype('float64')
            dst[dst<=0]=np.nan
            return dst
        if True:
            checkpoint = MID / 'threshold_pixel_indicators_checkpoint.csv'
            completed = set()
            if checkpoint.exists():
                prior = pd.read_csv(checkpoint)
                rows = prior.to_dict('records')
                completed = set(prior['GDLcode'].astype(str))
            for idx, rec in shp.iterrows():
                if str(rec.GDLcode) in completed:
                    continue
                geom=rec.geometry
                geom_mapping=geom.__geo_interface__ if geom is not None else None
                win=clipped_window(ec, geom)
                out={'GDLcode':rec.GDLcode}
                if win is None or geom is None or geom.is_empty:
                    out['status']='no_window'; rows.append(out); continue
                total=0.0; ec_weighted=0.0; n_valid=0
                gap_num={t:0.0 for t in LOWER}; deficit_num={t:0.0 for t in LOWER}
                excess_num={t:0.0 for t in UPPER}
                block=384
                for roff in range(int(win.row_off), int(win.row_off+win.height), block):
                    h=min(block, int(win.row_off+win.height)-roff)
                    for coff in range(int(win.col_off), int(win.col_off+win.width), block):
                        w=min(block, int(win.col_off+win.width)-coff)
                        tile=Window(coff,roff,w,h)
                        e=ec.read(1,window=tile,masked=True).astype('float64')
                        p=population_for_tile(tile)
                        tr=ec.window_transform(tile)
                        inside=geometry_mask([geom_mapping],out_shape=e.shape,transform=tr,invert=True,all_touched=False)
                        ev=np.asarray(e.filled(np.nan)); pv=np.asarray(p)
                        valid=inside & np.isfinite(ev) & np.isfinite(pv) & (pv>0) & (ev>=0)
                        if not valid.any(): continue
                        ecpc=ev[valid]/pv[valid]; weights=pv[valid]
                        total += float(weights.sum()); ec_weighted += float(np.sum(weights*ecpc)); n_valid += int(valid.sum())
                        for t in LOWER:
                            gap_num[t] += float(np.sum(weights*(ecpc<t)))
                            deficit_num[t] += float(np.sum(weights*np.maximum(0,1-ecpc/t)))
                        for t in UPPER:
                            excess_num[t] += float(np.sum(weights*(ecpc>t)))
                if total <= 0:
                    out['status']='no_valid_pixels'; rows.append(out); continue
                out.update(status='ok', pixel_population=float(total), n_ec_pixels=n_valid,
                           pixel_ecpc_mean_weighted=float(ec_weighted/total))
                for t in LOWER:
                    out[f'gap_{t}']=gap_num[t]/total
                    out[f'deficit_{t}']=deficit_num[t]/total
                for t in UPPER:
                    out[f'excess_{t}']=excess_num[t]/total
                rows.append(out)
                if len(rows)%50==0:
                    pd.DataFrame(rows).to_csv(checkpoint,index=False)
                    print('pixels',len(rows),'/',len(shp),flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(MID/'threshold_pixel_indicators_by_gdl.csv',index=False)
    return result


def weighted_mean(x,w):
    m=np.isfinite(x)&np.isfinite(w)&(w>0)
    return float(np.average(np.asarray(x)[m],weights=np.asarray(w)[m])) if m.any() else np.nan


def prepare_analysis(pixel):
    base=pd.read_csv(DATA_PATH)
    d=base.merge(pixel,on='GDLcode',how='left',validate='one_to_one')
    # Retain the already validated/published baseline indicators exactly; the
    # raster rerun supplies only the alternative-threshold scenarios.
    d['gap_300_rerun']=d['gap_300']
    d['deficit_300_rerun']=d['deficit_300']
    d['excess_5000_rerun']=d['excess_5000']
    d['gap_300']=d['sufficiency_gap']
    d['deficit_300']=d['sufficiency_deficit']
    d['excess_5000']=d['sufficiency_excess']
    d['income_clean']=d['income'].where(d['income']>=0,np.nan)
    d['wealth_clean']=d['wealth'].where(d['wealth']>=0,np.nan)
    d['SES_raw']=d['income_clean'].combine_first(d['wealth_clean'])
    d['female_share']=d['gender']
    d['GDP_pc']=d['GDP']/d['population'].replace(0,np.nan)
    d['log_GDP_pc']=np.log(d['GDP_pc'].where(d['GDP_pc']>0))
    d['log_population']=np.log(d['population'].where(d['population']>0))
    d.to_csv(MID/'threshold_analysis_base.csv',index=False)
    return d


def summaries(d):
    rows=[]
    for scope,region in [('Global','Global')]+[(r,r) for r in REGION_ORDER]:
        z=d if scope=='Global' else d[d.analytical_region==region]
        for t in LOWER:
            rows.append({'scope':scope,'threshold_type':'lower','threshold':t,
                         'gap':weighted_mean(z[f'gap_{t}'],z.population),
                         'deficit':weighted_mean(z[f'deficit_{t}'],z.population),
                         'excess':np.nan})
        for t in UPPER:
            rows.append({'scope':scope,'threshold_type':'upper','threshold':t,
                         'gap':np.nan,'deficit':np.nan,
                         'excess':weighted_mean(z[f'excess_{t}'],z.population)})
    s=pd.DataFrame(rows)
    s.to_csv(MID/'threshold_regional_full_summary.csv',index=False)
    return s


def stability(d):
    rows=[]
    for metric in ['gap','deficit']:
        ref=d[f'{metric}_300']
        for t in LOWER:
            cur=d[f'{metric}_{t}']; m=ref.notna()&cur.notna()
            rho=spearmanr(ref[m],cur[m]).statistic
            # Top quintile of GDL values, chosen to reduce degeneracy from mass points at 0/1.
            ref_hot=set(d.loc[m].nlargest(max(1,int(m.sum()*.2)),f'{metric}_300').GDLcode)
            cur_hot=set(d.loc[m].nlargest(max(1,int(m.sum()*.2)),f'{metric}_{t}').GDLcode)
            overlap=len(ref_hot&cur_hot)/len(ref_hot|cur_hot)
            rows.append({'indicator':metric,'threshold':t,'spearman_vs_300':rho,
                         'top20_hotspot_jaccard_vs_300':overlap,'n_gdl':int(m.sum())})
    ref=d['excess_5000']
    for t in UPPER:
        cur=d[f'excess_{t}']; m=ref.notna()&cur.notna()
        rows.append({'indicator':'excess','threshold':t,
                     'spearman_vs_300':spearmanr(ref[m],cur[m]).statistic,
                     'top20_hotspot_jaccard_vs_300':np.nan,'n_gdl':int(m.sum())})
    x=pd.DataFrame(rows).rename(columns={'spearman_vs_300':'spearman_vs_baseline'})
    x.to_csv(MID/'threshold_rank_stability.csv',index=False)
    return x


def fractional_models(d):
    predictors=['SES_raw','rural','education','female_share','log_GDP_pc','log_population','total_HDD','total_CDD']
    required=predictors+['iso_code','population']+[f'{m}_{t}' for m in ['gap','deficit'] for t in LOWER]
    sample=d.dropna(subset=required).copy()
    sample=sample[sample.population>0].copy()
    for v in predictors:
        sample['z_'+v]=(sample[v]-sample[v].mean())/sample[v].std(ddof=0)
    rhs=' + '.join('z_'+v for v in predictors)+' + C(iso_code)'
    rows=[]
    for metric in ['gap','deficit']:
        for t in LOWER:
            outcome=f'{metric}_{t}'
            fit=smf.glm(outcome+' ~ '+rhs,data=sample,family=sm.families.Binomial()).fit(
                cov_type='cluster',cov_kwds={'groups':sample.iso_code},maxiter=300)
            factor=float(np.mean(fit.predict(sample)*(1-fit.predict(sample))))
            ci=fit.conf_int()
            for v in ['SES_raw','rural','education','female_share']:
                p='z_'+v
                rows.append({'indicator':metric,'threshold':t,'variable':v,
                             'ame':float(fit.params[p]*factor),
                             'ame_low':float(ci.loc[p,0]*factor),'ame_high':float(ci.loc[p,1]*factor),
                             'p_value':float(fit.pvalues[p]),'n_gdl':len(sample),
                             'n_countries':sample.iso_code.nunique()})
    out=pd.DataFrame(rows)
    out.to_csv(MID/'threshold_fractional_logit_ame.csv',index=False)
    return out


def make_figure(s, st, ame):
    sns.set_theme(style='whitegrid',font='DejaVu Sans')
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.2),constrained_layout=True)
    # A: lower threshold response by region
    ax=axes[0,0]
    for r in REGION_ORDER:
        z=s[(s.scope==r)&(s.threshold_type=='lower')]
        ax.plot(z.threshold,z.gap,marker='o',lw=1.8,label=REGION_SHORT[r],color=REGION_COLORS[r])
    ax.axvline(300,color='0.25',ls='--',lw=1); ax.set(title='A  Lower-threshold response',xlabel='Lower threshold (kWh/cap/year)',ylabel='Below-floor population share')
    ax.legend(fontsize=7.5,ncol=2,frameon=False,loc='upper left')
    # B: upper response
    ax=axes[0,1]
    for r in REGION_ORDER:
        z=s[(s.scope==r)&(s.threshold_type=='upper')]
        ax.plot(z.threshold,z.excess,marker='o',lw=1.8,label=REGION_SHORT[r],color=REGION_COLORS[r])
    ax.axvline(5000,color='0.25',ls='--',lw=1); ax.set(title='B  Upper-threshold response',xlabel='Upper threshold (kWh/cap/year)',ylabel='Above-ceiling population share')
    # C rank robustness
    ax=axes[1,0]
    z=st[st.indicator.isin(['gap','deficit'])]
    for metric,color in [('gap','#B2182B'),('deficit','#D98C00')]:
        q=z[z.indicator==metric]; ax.plot(q.threshold,q.spearman_vs_baseline,marker='o',lw=2,label=metric.capitalize(),color=color)
    ax.axvline(300,color='0.25',ls='--',lw=1); ax.set_ylim(.7,1.01); ax.set(title='C  GDL rank stability',xlabel='Lower threshold (kWh/cap/year)',ylabel='Spearman correlation vs 300')
    ax.legend(frameon=False)
    # D AME stability, gap only for compactness
    ax=axes[1,1]
    labels={'SES_raw':'SES','rural':'Rurality','education':'Education','female_share':'Female share'}
    colors={'SES_raw':'#5AB4AC','rural':'#2166AC','education':'#8073AC','female_share':'#B2182B'}
    for v in labels:
        q=ame[(ame.indicator=='gap')&(ame.variable==v)]
        ax.plot(q.threshold,q.ame,marker='o',lw=1.8,label=labels[v],color=colors[v])
        ax.fill_between(q.threshold,q.ame_low,q.ame_high,color=colors[v],alpha=.10)
    ax.axhline(0,color='0.35',lw=.8); ax.axvline(300,color='0.25',ls='--',lw=1)
    ax.set(title='D  Social-gradient stability (Gap)',xlabel='Lower threshold (kWh/cap/year)',ylabel='Average marginal effect per 1 SD')
    ax.legend(fontsize=8,frameon=False,ncol=2)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'Figure_S_threshold_robustness.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def compact_table(s,st,ame):
    glob=s[s.scope=='Global'].copy()
    rows=[]
    for t in LOWER:
        r=glob[(glob.threshold_type=='lower')&(glob.threshold==t)].iloc[0]
        sg=st[(st.indicator=='gap')&(st.threshold==t)].iloc[0]
        sd=st[(st.indicator=='deficit')&(st.threshold==t)].iloc[0]
        ag=ame[(ame.indicator=='gap')&(ame.threshold==t)]
        signs='; '.join(f"{v}:{'+' if x>0 else '-'}" for v,x in zip(ag.variable,ag.ame))
        rows.append({'lower_threshold':t,'global_gap':r.gap,'global_deficit':r.deficit,
                     'rho_gap_vs_300':sg.spearman_vs_baseline,'rho_deficit_vs_300':sd.spearman_vs_baseline,
                     'hotspot_jaccard_gap':sg.top20_hotspot_jaccard_vs_300,'gap_ame_signs':signs})
    tab=pd.DataFrame(rows)
    tab.to_csv(OUT/'Table_S_threshold_robustness.csv',index=False)
    return tab


def main():
    cache=MID/'threshold_pixel_indicators_by_gdl.csv'
    pixel=pd.read_csv(cache) if cache.exists() else compute_pixel_thresholds()
    d=prepare_analysis(pixel)
    # Audit exact reproduction of the baseline threshold.
    audit=[]
    for a,b in [('gap_300_rerun','sufficiency_gap'),('deficit_300_rerun','sufficiency_deficit'),('excess_5000_rerun','sufficiency_excess')]:
        m=d[a].notna()&d[b].notna(); diff=(d.loc[m,a]-d.loc[m,b]).abs()
        audit.append({'new':a,'original':b,'n':int(m.sum()),'mae':float(diff.mean()),'max_abs_diff':float(diff.max()),'spearman':float(spearmanr(d.loc[m,a],d.loc[m,b]).statistic)})
    pd.DataFrame(audit).to_csv(MID/'baseline_reproduction_audit.csv',index=False)
    s=summaries(d); st=stability(d); ame=fractional_models(d)
    make_figure(s,st,ame); compact_table(s,st,ame)
    meta={'lower_thresholds':LOWER,'upper_thresholds':UPPER,'baseline_lower':300,'baseline_upper':5000,
          'input_data':str(DATA_PATH),'electricity_raster':str(EC_PATH),'population_raster':str(POP_PATH),
          'model':'Fractional logit M4 with country fixed effects and country-clustered standard errors',
          'outputs':['Figure_S_threshold_robustness.png','Table_S_threshold_robustness.csv']}
    (MID/'threshold_validation_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('DONE',OUT)

if __name__=='__main__': main()
