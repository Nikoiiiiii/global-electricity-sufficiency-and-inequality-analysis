# Global electricity sufficiency and inequality

Code, analysis-ready data, and selected results for research into the global spatial and social distribution of electricity sufficiency. The analysis uses a 2015 reference year and a primary sufficiency floor of 300 kWh per person per year. This repository is a working research archive; the paper citation and DOI will be added when available.

## Repository guide

| Location | Contents |
|---|---|
| `data/processed/` | Analysis-ready datasets, including sufficiency, controls, and national indicators. |
| `code/00_data_preparation/` | Original data-processing notebook. |
| `code/rq1/` | Electricity sufficiency distributions and inequality metrics. |
| `code/rq2/` | Spatial and social Theil decompositions. |
| `code/rq3/` | Social factors, fixed-effects models, interactions, diagnostics, and supplementary analysis. |
| `code/rq4/` | Contextual moderation models. |
| `code/rq4_sensitivity/` | Later country fixed-effects sensitivity analysis and plotting scripts. |
| `code/archive_pilot/` | Earlier exploratory analysis retained for provenance. |
| `results/` | Available main-analysis tables and the later RQ4 sensitivity outputs. |
| `validation/` | Available scripts, intermediate tables and supporting outputs for threshold, non-residential load, rurality and alternative metrics checks. |

## Data

The four CSVs in `data/processed/` represent successive analysis stages. RQ1 and RQ2 use `global_SE_EC_sufficiency_2015.csv`; RQ3 uses `global_SE_EC_sufficiency_controls_2015.csv`; RQ4 uses `global_SE_EC_sufficiency_controls_nationalindex_2015.csv`. The larger `global_SE_EC2015.csv.gz` is the earlier integrated dataset; decompress it to CSV before use. See [data/README.md](data/README.md) for the raw data and limitations.

## Running the analyses

The notebooks preserve the original research code; notebook outputs were cleared for sharing. They were originally written for a different folder structure and include absolute paths. **They are not yet configured to run directly from this repository**: first inspect and update input/output paths in each notebook or script, and obtain the source rasters and boundary files listed in the data notes. Some cells refer to intermediate files generated in earlier sessions. Python, Jupyter and the scientific and geospatial libraries listed in `requirements.txt` are required; exact historical versions have not been established.

A practical order is data preparation → RQ1 → RQ2 → RQ3 → RQ4 → validation. RQ3 model selection and diagnostics support the main model; `RQ3_final_results_packaging.ipynb` assembles outputs. Do not assume every validation result can be regenerated from this archive: scripts for some official-data checks have not yet been located. Execution and numerical reproduction have not yet been verified.

## Citation and licence

Publication citation, code licence, and dataset-specific licences are to be confirmed. Third-party datasets retain their original terms. Please contact the research team before reusing unpublished results.
