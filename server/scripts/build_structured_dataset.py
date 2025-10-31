import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def admissions(self) -> Path:
        return self.data_dir / "admissions.csv"

    @property
    def patients(self) -> Path:
        return self.data_dir / "patients.csv"

    @property
    def diagnoses_icd(self) -> Path:
        return self.data_dir / "diagnoses_icd.csv"

    @property
    def d_icd(self) -> Path:
        return self.data_dir / "d_icd_diagnoses.csv"

    @property
    def procedures_icd(self) -> Path:
        return self.data_dir / "procedures_icd.csv"

    @property
    def prescriptions(self) -> Path:
        return self.data_dir / "prescriptions.csv"

    @property
    def labevents(self) -> Path:
        return self.data_dir / "labevents.csv"

    @property
    def chartevents(self) -> Path:
        return self.data_dir / "chartevents.csv"

    @property
    def d_items(self) -> Path:
        return self.data_dir / "d_items.csv"

    @property
    def out_train(self) -> Path:
        return self.processed_dir / "structured_dataset.train.parquet"

    @property
    def out_val(self) -> Path:
        return self.processed_dir / "structured_dataset.val.parquet"

    @property
    def out_test(self) -> Path:
        return self.processed_dir / "structured_dataset.test.parquet"

    @property
    def out_schema(self) -> Path:
        return self.processed_dir / "structured_dataset.schema.json"

    @property
    def out_stats(self) -> Path:
        return self.project_root / "results" / "evaluation" / "patient_cohort_stats.json"


def ensure_dirs(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    (paths.project_root / "results" / "evaluation").mkdir(parents=True, exist_ok=True)


def read_csv_head(path: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def compute_age_at_admit(admittime: pd.Series, dob: pd.Series) -> pd.Series:
    days = (admittime - dob).dt.days
    years = np.floor_divide(days, 365).astype("Int64")
    # Ages < 0 can exist due to date shifting; clip at 0
    return years.clip(lower=0)


def filter_home_health_dispositions(adm: pd.DataFrame) -> pd.DataFrame:
    # Per project memory: focus on HOME and HOME HEALTH CARE dispositions
    # [[memory:10097589]]
    mask = adm["discharge_location"].isin(["HOME", "HOME HEALTH CARE"])
    return adm.loc[mask].copy()


def load_admissions(paths: Paths) -> pd.DataFrame:
    usecols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "discharge_location",
    ]
    df = pd.read_csv(paths.admissions, usecols=usecols, parse_dates=["admittime", "dischtime"], low_memory=False)
    df = filter_home_health_dispositions(df)
    # Remove rows with missing keys
    df = df.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"]).copy()
    return df


def load_patients(paths: Paths) -> pd.DataFrame:
    usecols = ["subject_id", "gender", "anchor_age", "dod"]
    # `patients.csv` in this snapshot uses anchor_age rather than dob; we approximate age at admit via anchor_age if dob unavailable
    df = pd.read_csv(paths.patients, usecols=usecols, parse_dates=["dod"], low_memory=False)
    return df


def attach_demographics(adm: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    merged = adm.merge(patients, on="subject_id", how="left")
    # Prefer calculating age from dob, but we only have anchor_age in this snapshot.
    # If anchor_age present, use it as a proxy for age_at_admit; otherwise leave null.
    if "anchor_age" in merged.columns:
        merged["age_at_admit"] = merged["anchor_age"].astype("Int64")
    else:
        merged["age_at_admit"] = pd.Series([pd.NA] * len(merged), dtype="Int64")
    merged = merged.drop(columns=[c for c in ["anchor_age"] if c in merged.columns])
    return merged


def load_diagnoses(paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dx = pd.read_csv(paths.diagnoses_icd, usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"], low_memory=False)
    dict_icd = pd.read_csv(paths.d_icd, usecols=["icd_code", "icd_version", "long_title"], low_memory=False)
    return dx, dict_icd


def filter_icd10(dx: pd.DataFrame) -> pd.DataFrame:
    return dx.loc[dx["icd_version"] == 10].copy()


def derive_primary_dx(dx10: pd.DataFrame) -> pd.DataFrame:
    # Primary diagnosis: the smallest seq_num per hadm_id
    # Some rows may lack seq_num; we handle by ranking nulls last
    dx10["_seq_rank"] = dx10["seq_num"].fillna(1e9)
    idx = dx10.sort_values(["hadm_id", "_seq_rank"]).groupby("hadm_id", as_index=False).head(1)
    primary = idx[["hadm_id", "icd_code"]].rename(columns={"icd_code": "primary_icd10"})
    primary = primary.drop_duplicates(subset=["hadm_id"])  # guard
    return primary


def aggregate_dx(dx10: pd.DataFrame, dict_icd: pd.DataFrame) -> pd.DataFrame:
    # Map descriptions
    dx10 = dx10.merge(dict_icd, on=["icd_code", "icd_version"], how="left")
    agg = (
        dx10.groupby("hadm_id")
        .agg(
            icd10_codes=("icd_code", lambda s: list(pd.unique(s.dropna()))),
            icd10_descriptions=("long_title", lambda s: list(pd.unique(s.dropna()))),
            num_icd10_codes=("icd_code", "nunique"),
        )
        .reset_index()
    )
    return agg


def simple_pdgm_bucket(primary_icd10: str) -> str:
    if not isinstance(primary_icd10, str) or primary_icd10 == "":
        return "Unknown"
    prefix = primary_icd10[:3].upper()
    if prefix.startswith("E11"):
        return "Endocrine"
    if prefix.startswith("I50"):
        return "Cardiac & Circulatory"
    if prefix.startswith("J44"):
        return "Respiratory"
    if prefix.startswith("L89"):
        return "Wounds"
    if prefix.startswith("I69"):
        return "Neuro/Rehab"
    if prefix.startswith("Z"):
        return "Surgical Aftercare"
    if prefix.startswith("I10"):
        return "Cardiac & Circulatory"
    return "Other"


def load_procedures(paths: Paths) -> pd.DataFrame:
    usecols = ["subject_id", "hadm_id", "icd_code", "icd_version"]
    proc = pd.read_csv(paths.procedures_icd, usecols=usecols, low_memory=False)
    return proc


def aggregate_procedures(proc: pd.DataFrame) -> pd.DataFrame:
    agg = (
        proc.groupby("hadm_id")
        .agg(procedures_icd10=("icd_code", lambda s: list(pd.unique(s.dropna()))), num_procedures_total=("icd_code", "nunique"))
        .reset_index()
    )
    return agg


def load_prescriptions(paths: Paths) -> pd.DataFrame:
    usecols = ["subject_id", "hadm_id", "starttime", "stoptime", "drug"]
    prs = pd.read_csv(paths.prescriptions, usecols=usecols, parse_dates=["starttime", "stoptime"], low_memory=False)
    return prs


def compute_discharge_like_meds(
    prs: pd.DataFrame, admissions_subset: pd.DataFrame
) -> pd.DataFrame:
    # Keep only rows within admission window
    merged = prs.merge(admissions_subset[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="inner")
    in_window = (merged["starttime"] <= merged["dischtime"]) & (merged["stoptime"].fillna(merged["dischtime"]) >= merged["admittime"])
    merged = merged.loc[in_window].copy()
    # Last order per drug near discharge (48h)
    near_discharge = (
        (merged["stoptime"].notna() & ((merged["dischtime"] - merged["stoptime"]).dt.total_seconds().abs() <= 48 * 3600))
        | merged["stoptime"].isna()
    )
    merged = merged.loc[near_discharge].copy()
    merged.sort_values(["hadm_id", "drug", "starttime"], inplace=True)
    last_per_drug = merged.groupby(["hadm_id", "drug"], as_index=False).tail(1)
    agg = (
        last_per_drug.groupby("hadm_id")
        .agg(meds_discharge_like=("drug", lambda s: list(pd.unique(s.dropna()))), medication_count=("drug", "nunique"))
        .reset_index()
    )
    return agg


def load_d_items(paths: Paths) -> Optional[pd.DataFrame]:
    if not paths.d_items.exists():
        return None
    return pd.read_csv(paths.d_items, usecols=["itemid", "label", "linksto", "category", "unitname"], low_memory=False)


def select_lab_itemids(d_items: Optional[pd.DataFrame]) -> Dict[str, Set[int]]:
    # Target analytes for specificity
    targets = {
        "Glucose": {"GLUCOSE"},
        "Creatinine": {"CREATININE"},
        "Sodium": {"SODIUM"},
        "Potassium": {"POTASSIUM"},
        "WBC": {"WBC", "WHITE BLOOD"},
        "Hemoglobin": {"HEMOGLOBIN", "HGB"},
    }
    if d_items is None:
        return {}
    itemid_map: Dict[str, Set[int]] = {k: set() for k in targets}
    upper_labels = d_items[["itemid", "label", "linksto"]].copy()
    upper_labels["LABEL_UP"] = upper_labels["label"].astype(str).str.upper()
    for analyte, tokens in targets.items():
        hits = upper_labels.loc[
            upper_labels["LABEL_UP"].apply(lambda x: any(token in x for token in tokens))
        ]
        itemid_map[analyte] = set(hits["itemid"].astype(int).tolist())
    return itemid_map


def iter_labevents_for_hadm(paths: Paths, hadm_ids: Set[int], use_itemids: Optional[Set[int]] = None, chunksize: int = 500_000) -> Iterable[pd.DataFrame]:
    cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    for chunk in pd.read_csv(paths.labevents, usecols=cols, parse_dates=["charttime"], chunksize=chunksize, low_memory=False):
        chunk = chunk.loc[chunk["hadm_id"].isin(hadm_ids)]
        if use_itemids:
            chunk = chunk.loc[chunk["itemid"].isin(list(use_itemids))]
        if not chunk.empty:
            yield chunk


def aggregate_labs_last48h(
    paths: Paths, admissions_subset: pd.DataFrame, d_items: Optional[pd.DataFrame]
) -> pd.DataFrame:
    hadm_ids: Set[int] = set(admissions_subset["hadm_id"].astype(int).tolist())
    analyte_to_itemids = select_lab_itemids(d_items)
    # Flatten all target itemids
    target_itemids: Set[int] = set()
    for s in analyte_to_itemids.values():
        target_itemids.update(s)

    frames: List[pd.DataFrame] = []
    for chunk in iter_labevents_for_hadm(paths, hadm_ids, use_itemids=target_itemids if target_itemids else None):
        merged = chunk.merge(admissions_subset[["hadm_id", "dischtime"]], on="hadm_id", how="inner")
        # Keep last 48h values
        within_48h = (merged["dischtime"] - merged["charttime"]).dt.total_seconds().between(0, 48 * 3600)
        merged = merged.loc[within_48h].copy()
        if not merged.empty:
            frames.append(merged)
    if not frames:
        return pd.DataFrame({"hadm_id": [], "labs_last48h": []})

    labs = pd.concat(frames, ignore_index=True)
    # Map itemid to analyte name, fallback to the numeric id string if unknown
    itemid_to_label: Dict[int, str] = {}
    for analyte, itemids in analyte_to_itemids.items():
        for iid in itemids:
            itemid_to_label[int(iid)] = analyte
    labs["analyte"] = labs["itemid"].map(itemid_to_label).fillna(labs["itemid"].astype(str))

    # Take last value per analyte
    labs.sort_values(["hadm_id", "analyte", "charttime"], inplace=True)
    last_vals = labs.groupby(["hadm_id", "analyte"], as_index=False).tail(1)

    def to_struct(group: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
        result: Dict[str, Dict[str, Optional[float]]] = {}
        for _, row in group.iterrows():
            result[str(row["analyte"]) ] = {"value": row["valuenum"] if pd.notna(row["valuenum"]) else None, "unit": row["valueuom"]}
        return result

    agg = (
        last_vals.groupby("hadm_id").apply(to_struct).reset_index(name="labs_last48h")
    )
    return agg


def build_dataset(
    paths: Paths,
    sample_target: int = 1500,
    seed: int = 17,
    skip_labs: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    # Load base tables
    admissions = load_admissions(paths)
    patients = load_patients(paths)
    admissions = attach_demographics(admissions, patients)
    admissions["length_of_stay_days"] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400.0

    # Diagnoses
    dx, dict_icd = load_diagnoses(paths)
    dx10 = filter_icd10(dx)
    # Keep only hadm_ids present in admissions subset to reduce memory
    dx10 = dx10.loc[dx10["hadm_id"].isin(set(admissions["hadm_id"].tolist()))].copy()
    # Code richness filter: >= 3 codes per hadm
    code_counts = dx10.groupby("hadm_id")["icd_code"].nunique().rename("num_icd10_codes")
    eligible_hadm = code_counts.loc[code_counts >= 3].index
    admissions = admissions.loc[admissions["hadm_id"].isin(set(map(int, eligible_hadm)))]

    # Recompute dx tables on filtered cohort
    dx10 = dx10.loc[dx10["hadm_id"].isin(set(admissions["hadm_id"].tolist()))].copy()
    primary = derive_primary_dx(dx10)
    agg_dx = aggregate_dx(dx10, dict_icd)
    epi = admissions.merge(agg_dx, on="hadm_id", how="inner")
    epi = epi.merge(primary, on="hadm_id", how="left")

    # Primary desc
    epi = epi.merge(
        dict_icd.rename(columns={"long_title": "primary_icd10_desc"})[["icd_code", "icd_version", "primary_icd10_desc"]]
        .rename(columns={"icd_code": "primary_icd10"}),
        on=["primary_icd10"],
        how="left",
    )

    # PDGM bucket
    epi["primary_pdgm_bucket_simple"] = epi["primary_icd10"].apply(simple_pdgm_bucket)

    # Procedures
    proc = load_procedures(paths)
    proc = proc.loc[proc["hadm_id"].isin(set(epi["hadm_id"].tolist()))].copy()
    agg_proc = aggregate_procedures(proc)
    epi = epi.merge(agg_proc, on="hadm_id", how="left")

    # Medications
    prs = load_prescriptions(paths)
    prs = prs.loc[prs["hadm_id"].isin(set(epi["hadm_id"].tolist()))].copy()
    meds = compute_discharge_like_meds(prs, epi[["hadm_id", "admittime", "dischtime"]])
    epi = epi.merge(meds, on="hadm_id", how="left")

    # Data sufficiency: ensure at least one of procedures, prescriptions, labs present
    has_proc = epi["num_procedures_total"].fillna(0) > 0
    has_meds = epi["medication_count"].fillna(0) > 0
    if skip_labs:
        has_labs = pd.Series(False, index=epi.index)
    else:
        # Labs (curated, last48h)
        d_items = load_d_items(paths)
        labs = aggregate_labs_last48h(paths, epi[["hadm_id", "dischtime"]], d_items)
        epi = epi.merge(labs, on="hadm_id", how="left")
        has_labs = epi["labs_last48h"].apply(lambda x: isinstance(x, dict) and len(x) > 0)

    epi = epi.loc[(has_proc | has_meds | has_labs)].copy()

    # Sample to target size with stratification by PDGM bucket
    rng = np.random.default_rng(seed)
    if len(epi) > sample_target:
        frames: List[pd.DataFrame] = []
        for bucket, dfb in epi.groupby("primary_pdgm_bucket_simple"):
            # proportional sampling
            n = int(np.round(sample_target * len(dfb) / len(epi)))
            if n == 0:
                n = min(1, len(dfb))
            idx = rng.choice(dfb.index.to_numpy(), size=min(n, len(dfb)), replace=False)
            frames.append(dfb.loc[idx])
        sampled = pd.concat(frames, ignore_index=True)
        # If rounding produced < target, top up from remaining
        if len(sampled) < sample_target:
            remain = epi.drop(sampled.index, errors="ignore")
            need = sample_target - len(sampled)
            if need > 0 and len(remain) > 0:
                idx2 = rng.choice(remain.index.to_numpy(), size=min(need, len(remain)), replace=False)
                sampled = pd.concat([sampled, remain.loc[idx2]], ignore_index=True)
        epi = sampled

    # Train/val/test split (70/15/15) stratified by PDGM bucket
    def stratified_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_parts: List[pd.DataFrame] = []
        val_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []
        rng_local = np.random.default_rng(seed)
        for _, g in df.groupby("primary_pdgm_bucket_simple"):
            idx = g.index.to_numpy()
            rng_local.shuffle(idx)
            n = len(idx)
            n_train = int(np.floor(0.70 * n))
            n_val = int(np.floor(0.15 * n))
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]
            train_parts.append(g.loc[train_idx])
            val_parts.append(g.loc[val_idx])
            test_parts.append(g.loc[test_idx])
        return (
            pd.concat(train_parts, ignore_index=True),
            pd.concat(val_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True),
        )

    train_df, val_df, test_df = stratified_split(epi, seed)

    # Stats for data quality report
    stats = {
        "num_rows": int(len(epi)),
        "by_pdgm_bucket": epi["primary_pdgm_bucket_simple"].value_counts().to_dict(),
        "demographics": {
            "gender_counts": epi["gender"].value_counts(dropna=False).to_dict(),
            "age_summary": epi["age_at_admit"].describe().to_dict(),
        },
        "modalities_presence": {
            "has_procedures": int(has_proc.sum()),
            "has_meds": int(has_meds.sum()),
            "has_labs": int(has_labs.sum()) if "has_labs" in locals() else 0,
        },
        "diagnoses": {
            "num_icd10_codes_summary": epi["num_icd10_codes"].describe().to_dict(),
            "top_primary_icd10": epi["primary_icd10"].value_counts().head(20).to_dict(),
        },
    }

    return train_df, val_df, test_df, stats


def to_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def write_schema(df: pd.DataFrame, schema_path: Path) -> None:
    cols = [
        {"name": c, "dtype": str(df[c].dtype)}
        for c in df.columns
    ]
    payload = {
        "columns": cols,
        "n_rows": int(len(df)),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    schema_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured episode-level dataset for synthetic summary generation")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--sample-target", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-labs", action="store_true", help="Skip lab aggregation (faster, v1.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    args = parser.parse_args()

    paths = Paths(project_root=Path(args.project_root))
    ensure_dirs(paths)

    if not args.overwrite and all(p.exists() for p in [paths.out_train, paths.out_val, paths.out_test, paths.out_schema]):
        print("Outputs already exist. Use --overwrite to regenerate.")
        return

    train_df, val_df, test_df, stats = build_dataset(paths, sample_target=args.sample_target, seed=args.seed, skip_labs=args.skip_labs)

    to_parquet_safe(train_df, paths.out_train)
    to_parquet_safe(val_df, paths.out_val)
    to_parquet_safe(test_df, paths.out_test)
    # Write schema using the union of columns from full set (train used as proxy)
    write_schema(pd.concat([train_df, val_df, test_df], ignore_index=True).iloc[:100], paths.out_schema)

    # Stats
    with open(paths.out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    print("Wrote:")
    print(f"  {paths.out_train}")
    print(f"  {paths.out_val}")
    print(f"  {paths.out_test}")
    print(f"  {paths.out_schema}")
    print(f"  {paths.out_stats}")


if __name__ == "__main__":
    main()


