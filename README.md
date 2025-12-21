# DuAL-Net

A dual-network framework for Alzheimer's Disease risk prediction integrating local and global genomic information.

## Overview

DuAL-Net combines:
- **Local Analysis**: Window-based SNP analysis using RF+TabNet ensemble
- **Global Analysis**: Annotation-guided scoring based on functional features

## Installation

```bash
git clone https://github.com/taehojo/DuAL-Net.git
cd DuAL-Net
pip install -r requirements.txt
```

## Usage

```bash
python dualnet.py --raw data.raw --dx diagnosis.txt --annotation annotations.csv --output results/
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--raw` | Raw SNP data file (.raw format) | Required |
| `--dx` | Diagnosis/phenotype file | Required |
| `--annotation` | SNP annotation CSV file | Required |
| `--output` | Output directory | output |
| `--alpha` | Integration weight (local vs global) | 0.6 |
| `--window` | Window size for local analysis | 100 |

### Input Format

- **Raw file**: PLINK .raw format
- **Diagnosis file**: Tab-separated with `New_Label` column (0=control, 1=case)
- **Annotation file**: CSV with `rs_id` column and binary annotation features

### Output

- `results.json`: AUC results for top/bottom SNP subsets
- `ranked_snps.csv`: SNPs ranked by combined score

## Web Server

Interactive web interface: [https://www.jolab.ai/dualnet](https://www.jolab.ai/dualnet)

## Citation

Jo T, et al. DuAL-Net: A Dual-Network Approach for Alzheimer's Disease Risk Prediction Integrating Local and Global Information in the APOE Region. (2025)

## License

MIT License
