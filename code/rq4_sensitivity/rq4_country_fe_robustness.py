from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA = Path('/Users/nico/Research/Energy sufficiency/Data/global_SE_EC_sufficiency_controls_nationalindex_2015.csv')
OUT = Path('rq4_country_fe_outputs')
OUT.mkdir(exist_ok=True)

SOCIAL = ['SES_raw', 'rural', 'education', 'female_share']
CONTEXT = ['national_GII', 'national_inequality_income', 'national_inequality_education', 'national_urbanisation_rate']
CONTROLS = ['log_GDP_pc', 'log_population', 'total_HDD', 'total_CDD']
INTERACTIONS = {
    'int_female_GII': ('female_share', 'national_GII'),
    'int_SES_income_ineq': ('SES_raw', 'national_inequality_income'),
    'int_education_edu_ineq': ('education', 'national_inequality_education'),
    'int_rural_urbanisation': ('rural', 'national_urbanisation_rate'),
}


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .1 else ''


def bh(p):
    p = pd.Series(p, dtype=float)
    order = p.sort_values().index
    ranked = p.loc[order].to_numpy()
    q = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = pd.Series(index=p.index, dtype=float)
    out.loc[order] = np.minimum(q, 1)
    return out


def add_fe(df, x):
    fe = pd.get_dummies(df['iso_code'], prefix='country', drop_first=True, dtype=float)
    return sm.add_constant(pd.concat([x.reset_index(drop=True), fe.reset_index(drop=True)], axis=1), has_constant='add')


def fit(df, columns, model_name):
    x = add_fe(df, df[columns].astype(float))
    y = df['log_sufficiency_ratio'].reset_index(drop=True)
    model = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': df['iso_code'].reset_index(drop=True), 'use_correction': True})
    rank = np.linalg.matrix_rank(x.to_numpy())
    sv = np.linalg.svd(x.to_numpy(), compute_uv=False)
    diag = {
        'model': model_name, 'n_gdl': len(df), 'n_countries': df.iso_code.nunique(),
        'parameters': x.shape[1], 'design_rank': rank, 'rank_deficiency': x.shape[1] - rank,
        'condition_number_raw_design': float(sv.max() / sv.min()),
        'r_squared': float(model.rsquared), 'adj_r_squared': float(model.rsquared_adj),
    }
    rows = []
    for term in INTERACTIONS:
        if term in model.params:
            rows.append({'model': model_name, 'term': term, 'estimate': model.params[term],
                         'clustered_se': model.bse[term], 'p_value': model.pvalues[term],
                         'ci_low': model.conf_int().loc[term, 0], 'ci_high': model.conf_int().loc[term, 1]})
    return model, pd.DataFrame(rows), diag


d = pd.read_csv(DATA)
d['income_clean'] = d['income'].where(d['income'].notna() & (d['income'] >= 0))
d['wealth_clean'] = d['wealth'].where(d['wealth'].notna() & (d['wealth'] >= 0))
d['SES_raw'] = d['income_clean'].combine_first(d['wealth_clean'])
d['female_share'] = d['gender']
d['GDP_pc'] = d['GDP'] / d['population'].where(d['population'] > 0)
d['log_GDP_pc'] = np.log(d['GDP_pc'].where(d['GDP_pc'] > 0))
d['log_population'] = np.log(d['population'].where(d['population'] > 0))
d['log_sufficiency_ratio'] = np.log(d['sufficiency_ratio'].where(d['sufficiency_ratio'] > 0))
required = ['log_sufficiency_ratio', 'iso_code', 'analytical_region'] + SOCIAL + CONTEXT + CONTROLS
s = d.loc[d[required].notna().all(axis=1)].copy().reset_index(drop=True)

# Reproduce the original global-complete-sample scaling exactly (population-unweighted, ddof=0).
std_rows = []
for v in SOCIAL + CONTEXT + CONTROLS:
    mu, sd = s[v].mean(), s[v].std(ddof=0)
    s['z_' + v] = (s[v] - mu) / sd
    std_rows.append({'variable': v, 'mean': mu, 'sd': sd})
for term, (a, b) in INTERACTIONS.items():
    s[term] = s['z_' + a] * s['z_' + b]

base_cols = ['z_' + v for v in SOCIAL + CONTROLS] + list(INTERACTIONS)
full, full_tab, full_diag = fit(s, base_cols, 'Country FE: four interactions')

# Focused model checks that the rurality result is not created by the other three interactions.
rural_cols = ['z_' + v for v in SOCIAL + CONTROLS] + ['int_rural_urbanisation']
focused, focused_tab, focused_diag = fit(s, rural_cols, 'Country FE: rurality interaction only')

# Within-country centring makes the lower-level rurality contrast explicit.
s['rural_wc'] = s['rural'] - s.groupby('iso_code')['rural'].transform('mean')
wc_sd = s['rural_wc'].std(ddof=0)
s['z_rural_wc'] = s['rural_wc'] / wc_sd
s['int_rural_wc_urbanisation'] = s['z_rural_wc'] * s['z_national_urbanisation_rate']
wc_cols = ['z_SES_raw', 'z_rural_wc', 'z_education', 'z_female_share'] + ['z_' + v for v in CONTROLS] + ['int_rural_wc_urbanisation']
wc, _, wc_diag = fit(s, wc_cols, 'Country FE: country-mean-centred rurality')
wc_tab = pd.DataFrame([{
    'model': 'Country FE: country-mean-centred rurality', 'term': 'int_rural_wc_urbanisation',
    'estimate': wc.params['int_rural_wc_urbanisation'], 'clustered_se': wc.bse['int_rural_wc_urbanisation'],
    'p_value': wc.pvalues['int_rural_wc_urbanisation'],
    'ci_low': wc.conf_int().loc['int_rural_wc_urbanisation', 0],
    'ci_high': wc.conf_int().loc['int_rural_wc_urbanisation', 1],
}])

table = pd.concat([full_tab, focused_tab, wc_tab], ignore_index=True)
mask = table.model.eq('Country FE: four interactions')
table.loc[mask, 'q_value_four_interactions'] = bh(table.loc[mask, 'p_value']).to_numpy()
table['significance'] = table.p_value.map(stars)
table.to_csv(OUT / 'Table_S_country_FE_cross_level_interactions.csv', index=False)

# Leave-one-country-out influence for the focal interaction (point estimates; clustered inference is in main table).
loo = []
for country in sorted(s.iso_code.unique()):
    sub = s.loc[s.iso_code.ne(country)].reset_index(drop=True)
    m, _, _ = fit(sub, rural_cols, 'loo')
    loo.append({'excluded_country': country, 'estimate': m.params['int_rural_urbanisation']})
loo = pd.DataFrame(loo)
loo['absolute_change_from_full'] = (loo.estimate - focused.params['int_rural_urbanisation']).abs()
loo.sort_values('absolute_change_from_full', ascending=False).to_csv(OUT / 'rurality_urbanisation_leave_one_country_out.csv', index=False)

# Diagnose the old NAO regional instability directly.
nao = s.loc[s.analytical_region.eq('Northern America and Oceania')].copy()
nao_country = nao.groupby('iso_code').agg(n_gdl=('GDLcode', 'size'), urbanisation=('national_urbanisation_rate', 'first'),
                                           education_inequality=('national_inequality_education', 'first'),
                                           rural_mean=('rural', 'mean'), rural_sd=('rural', 'std')).reset_index()
nao_country.to_csv(OUT / 'NAO_country_support_diagnostic.csv', index=False)

diagnostics = pd.DataFrame([full_diag, focused_diag, wc_diag])
diagnostics['within_country_rurality_sd'] = s['rural_wc'].std(ddof=0)
diagnostics['loo_min'] = loo.estimate.min()
diagnostics['loo_max'] = loo.estimate.max()
diagnostics.to_csv(OUT / 'country_FE_model_diagnostics.csv', index=False)
pd.DataFrame(std_rows).to_csv(OUT / 'standardization_parameters_reused.csv', index=False)

summary = {
    'input': str(DATA), 'n_gdl': len(s), 'n_countries': int(s.iso_code.nunique()),
    'full_rurality_interaction': table[(table.model == 'Country FE: four interactions') & (table.term == 'int_rural_urbanisation')].to_dict('records')[0],
    'focused_rurality_interaction': focused_tab.to_dict('records')[0],
    'within_centred_rurality_interaction': wc_tab.to_dict('records')[0],
    'loo_range': [float(loo.estimate.min()), float(loo.estimate.max())],
    'nao_countries': nao_country.to_dict('records'),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
