from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

PROJECT = Path('/Users/nico/Research/Energy sufficiency')
TABLE = PROJECT / 'Results/RQ4/Tables/Table_RQ4_6_plotting_global_region_interaction_coefficients.csv'
OUT = Path('rq4_country_fe_outputs/Figure_4_revised.png')

d = pd.read_csv(TABLE)
panels = [
    ('Female share x GII', 'A'),
    ('SES x income inequality', 'B'),
    ('Education x education inequality', 'C'),
    ('Rurality x urbanisation rate', 'D'),
]
unit_order = ['Global'] + list(d.loc[d.panel.eq(panels[0][0]), 'unit'].iloc[1:])
short = {
    'Sub-Saharan Africa':'Sub-Saharan Africa',
    'Northern Africa and Western Asia':'N. Africa & W. Asia',
    'South Asia':'South Asia',
    'Eastern and South-eastern Asia':'E. & SE Asia',
    'Europe':'Europe',
    'Latin America and the Caribbean':'Latin America & Caribbean',
    'Northern America and Oceania':'N. America & Oceania',
}

# A fixed symmetric scale prevents unstable estimates from stretching the legend.
COLOR_LIMIT = 4.0
norm = TwoSlopeNorm(vmin=-COLOR_LIMIT, vcenter=0, vmax=COLOR_LIMIT)
cmap = plt.get_cmap('RdBu_r')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.titlesize':16,
                     'axes.labelsize':13,'xtick.labelsize':10.5,'ytick.labelsize':10.5})
fig, axes = plt.subplots(2,2,figsize=(16.5,11.8))

for ax,(panel,letter) in zip(axes.ravel(),panels):
    sub=d.loc[d.panel.eq(panel)].copy()
    sub['order']=sub.unit.map({u:i for i,u in enumerate(unit_order)})
    sub=sub.sort_values('order')
    sub['lo']=sub.coef-1.96*sub.se
    sub['hi']=sub.coef+1.96*sub.se
    # Estimate or interval too extreme to display without destroying the useful scale.
    sub['out_of_range']=(sub.coef.abs()>COLOR_LIMIT) | ((sub.hi-sub.lo)>16)
    stable=sub.loc[~sub.out_of_range]
    lo,hi=stable.lo.min(),stable.hi.max()
    width=max(hi-lo,.5)
    xmin,xmax=lo-.17*width,hi+.24*width
    ax.set_xlim(xmin,xmax)
    ax.axvline(0,color='.35',ls='--',lw=1.0)
    ax.grid(axis='x',alpha=.17)
    y=np.arange(len(sub))[::-1]

    for yy,(_,row) in zip(y,sub.iterrows()):
        is_global=row.unit=='Global'
        # Clip only colour saturation, never the underlying reported estimate.
        color='#303030' if is_global else cmap(norm(np.clip(row.coef,-COLOR_LIMIT,COLOR_LIMIT)))
        if row.out_of_range:
            left=row.coef<0
            edge=xmin+.015*width if left else xmax-.015*width
            ax.scatter(edge,yy,marker='<' if left else '>',s=105,facecolor=color,
                       edgecolor='white',linewidth=1.3,zorder=4,clip_on=False)
            ax.text(edge+(.04*width if left else -.04*width),yy+.18,'Out of range',
                    ha='left' if left else 'right',va='center',fontsize=9.7,
                    color='.28',fontweight='bold')
            continue

        nonsig=str(row.stars) in ('','nan')
        line_color='#aebbc2' if (nonsig and not is_global) else color
        ax.hlines(yy,max(row.lo,xmin),min(row.hi,xmax),color=line_color,lw=2.25,alpha=.95)
        hollow='high risk' in str(row.warning)
        ax.scatter(row.coef,yy,s=92 if is_global else 82,
                   facecolor='white' if hollow else color,
                   edgecolor=color if hollow else 'white',linewidth=2 if hollow else 1.2,zorder=4)
        if not nonsig:
            right=(xmax-row.coef)>=(row.coef-xmin)
            xx=min(row.hi+.035*width,xmax-.025*width) if right else max(row.lo-.035*width,xmin+.025*width)
            ha='left' if right else 'right'
            ax.text(xx,yy+.17,str(row.stars),color='#c9252d',fontsize=15,
                    fontweight='bold',ha=ha,va='center')
            ax.text(xx,yy-.13,f"{row.coef:+.2f}, q={row.q_value:.3f}",color='.15',
                    fontsize=10.5,fontweight='bold',ha=ha,va='center')

    labels=[]
    for _,row in sub.iterrows():
        if row.unit=='Global': labels.append('Global')
        else:
            risk=', high risk' if 'high risk' in str(row.warning) else ', caution' if 'caution' in str(row.warning) else ''
            labels.append(f"{short.get(row.unit,row.unit)}\n(n={int(row.n_countries)}{risk})")
    ax.set_yticks(y,labels)
    ax.set_title(panel.replace(' x ',' × '),fontweight='bold',pad=11)
    ax.text(-.12,1.07,letter,transform=ax.transAxes,fontsize=17,fontweight='bold',va='top')
    ax.set_xlabel('Interaction coefficient')

fig.suptitle('Contextual moderation of subnational social effects',fontsize=22.5,fontweight='bold',y=.995)
smap=ScalarMappable(norm=norm,cmap=cmap); smap.set_array([])
cax=fig.add_axes([.34,.058,.32,.018])
cb=fig.colorbar(smap,cax=cax,orientation='horizontal',ticks=[-4,-2,0,2,4])
cb.set_label('Interaction coefficient (colour capped at ±4)',fontsize=10.5,labelpad=2)
cb.ax.tick_params(labelsize=9)
fig.text(.5,.008,'Points show interaction coefficients; horizontal lines show 95% confidence intervals. '
                  'Out-of-range estimates are shown at the boundary using the most saturated colour.',
         ha='center',fontsize=10.5)
fig.subplots_adjust(left=.19,right=.985,top=.90,bottom=.13,hspace=.34,wspace=.34)
fig.savefig(OUT,dpi=300,bbox_inches='tight',facecolor='white')
print(OUT.resolve())
