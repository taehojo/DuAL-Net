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
