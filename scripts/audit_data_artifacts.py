import argparse
import json
import os
from typing import List, Optional

import pandas as pd


NER_DATASET_PATH = \
    "/Users/benjamindykstra/development/icd-10-coding/data/processed/structured_dataset_with_discharge_summaries_ner_features.train.csv"
RETRIEVAL_RESULTS_PATH = \
    "/Users/benjamindykstra/development/icd-10-coding/results/evaluation/retrieval_strategies_comparison.csv"
ICD10_DESCRIPTIONS_CSV = \
    "/Users/benjamindykstra/development/icd-10-coding/data/idc10_links/icd10_code_descriptions.csv"
ICD10_DESCRIPTIONS_PARQUET = \
    "/Users/benjamindykstra/development/icd-10-coding/data/idc10_links/icd10_code_descriptions.parquet"
MIMIC_D_ICD_DIAGNOSES = \
    "/Users/benjamindykstra/development/icd-10-coding/data/d_icd_diagnoses.csv"
COHORT_STATS_JSON = \
    "/Users/benjamindykstra/development/icd-10-coding/results/evaluation/patient_cohort_stats.json"


def safe_read_csv(path: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


def safe_read_parquet(path: str, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return None


def audit_ner_dataset(path: str) -> dict:
    df_head = safe_read_csv(path, nrows=5)
    if df_head is None:
        return {"exists": False}
    result = {
        "exists": True,
        "path": path,
        "columns": list(df_head.columns),
        "sample_row_0_keys": list(df_head.iloc[0].to_dict().keys()),
        "has_columns": {},
    }
    expected_cols = [
        "subject_id",
        "hadm_id",
        "discharge_summary",
        "diagnosis_codes",
        "true_icd_codes",
        "missing_codes",
        "disease_entities",
        "drug_entities",
        "anatomy_entities",
        "primary_pdgm_bucket_simple",
    ]
    for c in expected_cols:
        result["has_columns"][c] = c in df_head.columns
    return result


def audit_retrieval_results(path: str) -> dict:
    df = safe_read_csv(path, nrows=5)
    if df is None:
        return {"exists": False}
    return {
        "exists": True,
        "path": path,
        "columns": list(df.columns),
        "strategies_head": df["strategy"].head(5).tolist() if "strategy" in df.columns else [],
    }


def audit_icd_descriptions(csv_path: str, parquet_path: str, mimic_diag_path: str) -> dict:
    csv_df = safe_read_csv(csv_path, nrows=5)
    pq_df = safe_read_parquet(parquet_path, columns=None)
    if pq_df is not None:
        pq_df = pq_df.head(5)
    mimic_df = safe_read_csv(mimic_diag_path, nrows=5)

    return {
        "csv": {
            "exists": csv_df is not None,
            "path": csv_path,
            "columns": list(csv_df.columns) if csv_df is not None else [],
        },
        "parquet": {
            "exists": pq_df is not None,
            "path": parquet_path,
            "columns": list(pq_df.columns) if pq_df is not None else [],
        },
        "mimic_d_icd_diagnoses": {
            "exists": mimic_df is not None,
            "path": mimic_diag_path,
            "columns": list(mimic_df.columns) if mimic_df is not None else [],
        },
    }


def audit_cohort_stats(path: str) -> dict:
    if not os.path.exists(path):
        return {"exists": False}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {
            "exists": True,
            "keys": list(data.keys()),
            "by_pdgm_bucket_keys": list(data.get("by_pdgm_bucket", {}).keys()),
            "num_rows": data.get("num_rows"),
        }
    except Exception:
        return {"exists": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit available data artifacts for missing-code detection plan.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    ner = audit_ner_dataset(NER_DATASET_PATH)
    retrieval = audit_retrieval_results(RETRIEVAL_RESULTS_PATH)
    icd = audit_icd_descriptions(ICD10_DESCRIPTIONS_CSV, ICD10_DESCRIPTIONS_PARQUET, MIMIC_D_ICD_DIAGNOSES)
    cohort = audit_cohort_stats(COHORT_STATS_JSON)

    summary = {
        "ner_dataset": ner,
        "retrieval_results": retrieval,
        "icd_descriptions": icd,
        "cohort_stats": cohort,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("NER dataset:")
    print(json.dumps(ner, indent=2))
    print("\nRetrieval results:")
    print(json.dumps(retrieval, indent=2))
    print("\nICD descriptions:")
    print(json.dumps(icd, indent=2))
    print("\nCohort stats:")
    print(json.dumps(cohort, indent=2))


if __name__ == "__main__":
    main()


