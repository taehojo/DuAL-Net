# DuAL-Net: Alzheimer's Prediction Framework

## Overview

DuAL-Net is a framework designed for Alzheimer's Disease (AD) risk prediction using Whole Genome Sequencing (WGS) data. It integrates two analysis approaches:

1.  **Local Analysis:** Examines genomic context using non-overlapping windows of Single Nucleotide Polymorphisms (SNPs).
2.  **Global Analysis:** Utilizes functional SNP annotations to capture broader genomic relationships.

The framework employs an ensemble model combining TabNet and Random Forest classifiers with a Logistic Regression meta-learner. It generates AD risk predictions and ranks SNPs based on their contribution to the prediction.

## Web Server

A web-based implementation is accessible at:
**[https://www.jolab.ai/dualnet](https://www.jolab.ai/dualnet)**

## Local Execution

### Project Structure
```
DuAL-Net/
├── scripts/
│   ├── DuAL-Net.py      # Nested CV & SNP ranking
│   └── validating.py    # External validation
├── results/
│   ├── ADNI_nested_cv_results.json
│   ├── ADSP_ADC_validation_results.json
│   └── consensus_snp_rankings.csv
├── requirements.txt
└── README.md
```

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/taehojo/DuAL-Net.git
    cd DuAL-Net
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Step 1: DuAL-Net.py (Nested CV & SNP Ranking)

Runs 5×5 nested cross-validation on discovery cohort and generates SNP rankings.

```bash
python scripts/DuAL-Net.py \
  --raw <genotype.raw> \
  --dx <phenotype.txt> \
  --anno <annotation.csv> \
  --output ./results
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--raw` | PLINK .raw genotype file | Required |
| `--dx` | Phenotype file with `New_Label` column | Required |
| `--anno` | SNP annotation CSV with `rs_id` column | Required |
| `--output` | Output directory | `./output_modeling` |
| `--n_outer` | Outer CV folds | 5 |
| `--n_inner` | Inner CV folds | 5 |
| `--window_size` | SNP window size | 100 |
| `--top_n` | Number of top SNPs to select | 100 |
| `--alpha_step` | Alpha search step size | 0.1 |

**Output:**
- `consensus_snp_rankings.csv` - SNP rankings aggregated across folds
- `modeling_results.json` - AUC, accuracy, and optimal alpha
- `fold_*/merged_scores.csv` - Per-fold SNP scores

#### Step 2: validating.py (External Validation)

Validates SNP rankings on an independent cohort by comparing top-ranked vs bottom-ranked SNPs.

```bash
python scripts/validating.py \
  --rankings ./results/consensus_snp_rankings.csv \
  --raw <validation_genotype.raw> \
  --anno <annotation.csv> \
  --output ./validation
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--rankings` | SNP rankings CSV from Step 1 | Required |
| `--raw` | Validation cohort .raw file | Required |
| `--anno` | Annotation file (for rsID to position mapping) | Optional |
| `--output` | Output directory | `./output_validation` |
| `--top_n` | Comma-separated list of N values | `100,500,1000` |

**Output:**
- `validation_results.json` - AUC for top/bottom SNP subsets

### Input Data Format

| File | Format | Description |
|------|--------|-------------|
| Genotype | PLINK `.raw` | Additive coding (0/1/2) |
| Phenotype | Tab-delimited | Must contain `New_Label` column (0=control, 1=case) |
| Annotation | CSV | Must contain `rs_id` column and numeric annotation columns |

## Pre-computed Results

The `results/` folder contains pre-computed results from the manuscript:

| File | Description |
|------|-------------|
| `ADNI_nested_cv_results.json` | ADNI discovery cohort (n=1,050): AUC=0.698, α=0.66 |
| `ADSP_ADC_validation_results.json` | ADSP ADC validation cohort (n=5,570): Top vs Bottom SNP comparison |
| `consensus_snp_rankings.csv` | SNP rankings aggregated across 5 folds |

## Citation

* Citation information will be added soon.

## License

* © Dr. Jo's Medical AI Research lab, IUSM | www.jolab.ai
