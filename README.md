# DuAL-Net: Alzheimer's Prediction Framework

## Overview

DuAL-Net is a framework designed for Alzheimer's Disease (AD) risk prediction using Whole Genome Sequencing (WGS) data. It integrates two analysis approaches:

1.  **Local Analysis:** Examines genomic context using non-overlapping windows of Single Nucleotide Polymorphisms (SNPs).
2.  **Global Analysis:** Utilizes functional SNP annotations to capture broader genomic relationships.

The framework employs an ensemble model combining TabNet and Random Forest classifiers. It generates AD risk predictions and ranks SNPs based on their contribution to the prediction.

## Web Server

A web-based implementation is accessible at:
**[https://www.jolab.ai/dualnet](https://www.jolab.ai/dualnet)**

## Local Execution

### Project Structure
```
DuAL-Net/
├── scripts/
│   ├── DuAL-Net.py           # Main modeling script
│   ├── validating.py         # External validation script
│   ├── annotate_snps.py
│   └── run_dual_net_analysis.py
├── src/
│   ├── utils.py
│   ├── annotation_transformer.py
│   ├── modeling.py
│   ├── analysis.py
│   └── plotting.py
├── data/
├── results/
├── run_pipeline.sh
├── requirements.txt
└── README.md
```

### Setup

1.  **Get the Code:**
    Clone the repository using Git:
    ```bash
    git clone https://github.com/taehojo/DuAL-Net.git
    cd DuAL-Net
    ```

2.  **Install Dependencies:**
    Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Ensembl Data:**
    Download Ensembl data required for SNP annotation by `pyensembl`:
    ```bash
    pyensembl install --release 108 --species homo_sapiens
    ```

### Running the Pipeline

1.  **Configure Script:**
    * Open `run_pipeline.sh` in a text editor.
    * Update the file paths (`INPUT_BIM`, `RAW_DATA`, `DX_DATA`, `ANNOTATION_CSV`) and output settings (`PREFIX`, `RESULT_DIR`) to match your environment.
    * Adjust analysis parameters (e.g., `WINDOW_SIZE`) if necessary.

2.  **Execute Pipeline:**
    Run the script from the terminal:
    ```bash
    bash run_pipeline.sh
    ```
    * The script executes two main stages: SNP annotation followed by the DuAL-Net analysis (local/global processing, scoring, ranking).
    * Results are saved to the specified `RESULT_DIR`.

### Standalone Scripts

For reproducibility, two standalone scripts are provided:

#### Step 1: DuAL-Net.py (Nested CV & SNP Ranking)

Runs nested cross-validation on discovery cohort and generates SNP rankings.

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

**Output:** `consensus_snp_rankings.csv`

#### Step 2: validating.py (External Validation)

Validates SNP rankings on an independent cohort.

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
| `--anno` | Annotation file (for rsID mapping) | Optional |
| `--output` | Output directory | `./output_validation` |
| `--top_n` | Comma-separated list of N values | `100,500,1000` |

**Output:** `validation_results.json`

---

### Input Data Format

Required input files:

* **SNP List (`.bim`):** Standard PLINK `.bim` format (for annotation).
* **Raw Genetic Data (`.raw`):** Standard PLINK `.raw` format (for analysis).
* **Phenotype/Diagnosis Data (`.txt`):** Text file matching samples in `.raw` file, containing a phenotype column (0/1 for control/case).

### Output Files

Key output files generated in the results directory:

* `{PREFIX}_combined_results.csv`: Final SNP ranking with scores.
* `{PREFIX}_roc_curves.pdf`: ROC curve plots for performance evaluation.
* `{PREFIX}_analysis_logs.txt`: Log file of the analysis run.
* Intermediate result files (e.g., `_local_results.csv`, `_global_results.csv`).
* The annotation CSV file (e.g., `data/snp_annotations.csv`).

## Citation

* Citation information will be added soon.

## License

* © Dr. Jo's Medical AI Research lab, IUSM | www.jolab.ai
