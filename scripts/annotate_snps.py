#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pyensembl
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ENSEMBL_RELEASE = 108
ENSEMBL_SPECIES = "homo_sapiens"
ENSEMBL_REST_SERVER = "https://rest.ensembl.org"
REQUEST_HEADERS = {"Content-Type": "application/json"}
MAX_WORKERS = 4
RETRY_DELAY = 5
MAX_RETRIES = 3

def safe_request(url, headers, retries=MAX_RETRIES, delay=RETRY_DELAY):
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Request failed (attempt {attempt+1}/{retries+1}): {url}. Error: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for {url}. Returning None.")
                return None

def get_snp_info(rs_id):
    ext = f"/variation/{ENSEMBL_SPECIES}/{rs_id}?"
    data = safe_request(ENSEMBL_REST_SERVER + ext, REQUEST_HEADERS)
    if data:
        return {
            'most_severe_consequence': data.get('most_severe_consequence', 'Unknown'),
            'clinical_significance': data.get('clinical_significance', [])
        }
    return {'most_severe_consequence': 'Unknown', 'clinical_significance': []}

def get_gene_info(ensembl_db, chrom, pos):
    try:
        genes = ensembl_db.genes_at_locus(contig=str(chrom), position=int(pos))
        if genes:
            protein_coding = [g for g in genes if g.biotype == 'protein_coding']
            target_gene = protein_coding[0] if protein_coding else genes[0]
            return target_gene.gene_name, target_gene.biotype
    except Exception as e:
        print(f"Warning: pyensembl error in get_gene_info for {chrom}:{pos}: {str(e)}")
    return None, None

def get_regulatory_features(chrom, pos):
    pos = int(pos)
    ext = f"/overlap/region/{ENSEMBL_SPECIES}/{chrom}:{pos}-{pos}?feature=regulatory"
    data = safe_request(ENSEMBL_REST_SERVER + ext, REQUEST_HEADERS)
    return [feat.get('feature_type', 'Unknown') for feat in data] if data else []

def get_epigenetic_info(chrom, pos):
    pos = int(pos)
    ext = f"/overlap/region/{ENSEMBL_SPECIES}/{chrom}:{pos}-{pos}?feature=motif;feature=epigenomic"
    data = safe_request(ENSEMBL_REST_SERVER + ext, REQUEST_HEADERS)
    return [feat.get('feature_type', 'Unknown') for feat in data] if data else []

def get_exon_intron_utr(ensembl_db, chrom, pos):
    try:
        pos = int(pos)
        transcripts = ensembl_db.transcripts_at_locus(contig=str(chrom), position=pos)
        if transcripts:
            is_exonic, is_intronic, is_utr = False, False, False
            coding_transcripts = [t for t in transcripts if t.biotype == 'protein_coding']
            target_transcripts = coding_transcripts if coding_transcripts else transcripts

            for transcript in target_transcripts:
                if any(exon.start <= pos <= exon.end for exon in transcript.exons):
                    is_exonic = True
                    is_utr_5 = hasattr(transcript, 'five_prime_utr_intervals') and any(start <= pos <= end for start, end in transcript.five_prime_utr_intervals)
                    is_utr_3 = hasattr(transcript, 'three_prime_utr_intervals') and any(start <= pos <= end for start, end in transcript.three_prime_utr_intervals)
                    if is_utr_5 or is_utr_3:
                        is_utr = True
                    break

            if not is_exonic:
                for transcript in target_transcripts:
                    if transcript.start <= pos <= transcript.end:
                         is_intronic_in_this = False
                         for i in range(len(transcript.exon_intervals) - 1):
                             intron_start = transcript.exon_intervals[i][1] + 1
                             intron_end = transcript.exon_intervals[i+1][0] - 1
                             if intron_start <= pos <= intron_end:
                                 is_intronic_in_this = True
                                 break
                         if is_intronic_in_this:
                            is_intronic = True
                            break
            return is_exonic, is_intronic, is_utr

    except Exception as e:
        print(f"Warning: pyensembl error in get_exon_intron_utr for {chrom}:{pos}: {str(e)}")
    return None, None, None

def get_conservation_score(chrom, pos):
    pos = int(pos)
    ext = f"/conservation/region/{ENSEMBL_SPECIES}/{chrom}:{pos}-{pos}?"
    data = safe_request(ENSEMBL_REST_SERVER + ext, REQUEST_HEADERS)
    if data and isinstance(data, list) and data[0]:
        return data[0].get('scores', [{}])[0].get('value') if data[0].get('scores') else data[0].get('conservation_score')
    return None

def process_variant(variant_data, ensembl_db):
    row_dict = variant_data.to_dict()
    rs_id = row_dict['rs_id']
    chrom = str(row_dict['chrom'])
    pos = int(row_dict['pos'])

    snp_info = get_snp_info(rs_id)
    gene_name, gene_biotype = get_gene_info(ensembl_db, chrom, pos)
    regulatory_features = get_regulatory_features(chrom, pos)
    epigenetic_features = get_epigenetic_info(chrom, pos)
    exon_intron_utr = get_exon_intron_utr(ensembl_db, chrom, pos)
    conservation_score = get_conservation_score(chrom, pos)

    clinical_sig_columns = [
        'uncertain_significance', 'likely_pathogenic', 'pathogenic', 'drug_response', 'other',
        'risk_factor', 'association', 'protective', 'established_risk_allele'
    ]
    clinical_sig_dict = {col: 0 for col in clinical_sig_columns}
    if snp_info['clinical_significance']:
        normalized_fetched = {term.lower().replace(' ', '_') for term in snp_info['clinical_significance']}
        for col in clinical_sig_columns:
            if col in normalized_fetched:
                clinical_sig_dict[col] = 1

    result = {
        'rs_id': rs_id,
        'chrom': chrom,
        'pos': pos,
        'allele1': row_dict['allele1'],
        'allele2': row_dict['allele2'],
        'most_severe_consequence': snp_info['most_severe_consequence'],
        'gene_name': gene_name,
        'gene_biotype': gene_biotype,
        'regulatory_features': regulatory_features,
        'epigenetic_features': epigenetic_features,
        'exon_intron_utr': exon_intron_utr,
        'conservation_score': conservation_score,
        **clinical_sig_dict
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.bim> <output.csv>")
        sys.exit(1)

    input_bim_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    print(f"Initializing Ensembl Release {ENSEMBL_RELEASE} for {ENSEMBL_SPECIES}...")
    try:
        ensembl = pyensembl.EnsemblRelease(release=ENSEMBL_RELEASE, species=ENSEMBL_SPECIES)
        print(f"Using Ensembl release: {ensembl.release}")
        print(f"Genome assembly: {ensembl.reference_name}")
    except Exception as e:
        print(f"Error initializing Ensembl: {str(e)}")
        print(f"Please ensure Ensembl data is installed:")
        print(f"  pyensembl install --release {ENSEMBL_RELEASE} --species {ENSEMBL_SPECIES}")
        sys.exit(1)

    print(f"Loading BIM file: {input_bim_path}")
    try:
        bim_columns = ['chrom', 'rs_id', 'cm', 'pos', 'allele1', 'allele2']
        variants_df = pd.read_csv(input_bim_path, delim_whitespace=True, header=None, names=bim_columns,
                                  dtype={'chrom': str})
        print(f"Loaded {len(variants_df)} variants.")

        variants_df = variants_df[variants_df['rs_id'].str.startswith('rs', na=False)].copy()
        variants_df = variants_df[['rs_id', 'chrom', 'pos', 'allele1', 'allele2']]
        print(f"Filtered down to {len(variants_df)} variants with rsIDs.")

        variants_df = variants_df[pd.to_numeric(variants_df['pos'], errors='coerce').notnull()]
        variants_df['pos'] = variants_df['pos'].astype(int)
        print(f"Kept {len(variants_df)} variants with valid positions.")

    except FileNotFoundError:
        print(f"Error: Input BIM file not found at {input_bim_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing BIM file: {e}")
        sys.exit(1)

    if variants_df.empty:
        print("No valid variants found in the BIM file after filtering. Exiting.")
        pd.DataFrame(columns=[
             'rs_id', 'chrom', 'pos', 'allele1', 'allele2', 'most_severe_consequence',
             'gene_name', 'gene_biotype', 'regulatory_features', 'epigenetic_features',
             'exon_intron_utr', 'conservation_score', 'uncertain_significance',
             'likely_pathogenic', 'pathogenic', 'drug_response', 'other', 'risk_factor',
             'association', 'protective', 'established_risk_allele'
        ]).to_csv(output_csv_path, index=False)
        print(f"Empty annotation file created at '{output_csv_path}'.")
        sys.exit(0)

    print(f"Annotating {len(variants_df)} variants using up to {MAX_WORKERS} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_variant, variant_row, ensembl): index
                   for index, variant_row in variants_df.iterrows()}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating Variants"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                index = futures[future]
                print(f"Error processing variant at index {index}: {e}")

    print(f"\nCollected results for {len(results)} variants.")
    if not results:
        print("Warning: No results were successfully collected after annotation.")

    annotated_variants_df = pd.DataFrame(results)

    try:
        annotated_variants_df.to_csv(output_csv_path, index=False)
        print(f"Annotation complete. Results saved to '{output_csv_path}'.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        sys.exit(1)
