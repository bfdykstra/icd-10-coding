#!/usr/bin/env python3
import os
import argparse
import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import pandas as pd
from pydantic import BaseModel
import instructor
from openai import AsyncOpenAI


@dataclass
class Config:
    in_path: str
    out_path: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    concurrency: int = 12
    max_retries: int = 5
    sample: int | None = None


class DischargeSummary(BaseModel):
    discharge_summary: str
    diagnosis_codes: List[str]
    policy: Literal["primary_only", "partial", "all"]


def choose_policy(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return "primary_only"
    if r < 0.8:
        return "partial"
    return "all"


def format_list(values: List[Any], limit: int = 10) -> str:
    if not values:
        return "None"
    vs = [str(v) for v in values[:limit]]
    if len(values) > limit:
        vs.append("…")
    return ", ".join(vs)


def build_messages(row: pd.Series, policy: str) -> List[Dict[str, str]]:
    subject_id = int(row.get("subject_id")) if pd.notna(row.get("subject_id")) else None
    hadm_id = int(row.get("hadm_id")) if pd.notna(row.get("hadm_id")) else None
    age = int(row.get("age_at_admit")) if pd.notna(row.get("age_at_admit")) else None
    gender = str(row.get("gender")) if pd.notna(row.get("gender")) else None
    los_days = float(row.get("length_of_stay_days")) if pd.notna(row.get("length_of_stay_days")) else None
    admission_type = str(row.get("admission_type")) if pd.notna(row.get("admission_type")) else None

    primary_icd10 = row.get("primary_icd10") if pd.notna(row.get("primary_icd10")) else None
    primary_icd10_desc = row.get("primary_icd10_desc") if pd.notna(row.get("primary_icd10_desc")) else None
    icd10_codes = row.get("icd10_codes") if isinstance(row.get("icd10_codes"), list) else []
    icd10_descs = row.get("icd10_descriptions") if isinstance(row.get("icd10_descriptions"), list) else []

    procedures = row.get("procedures_icd10") if isinstance(row.get("procedures_icd10"), list) else []
    meds = row.get("meds_discharge_like") if isinstance(row.get("meds_discharge_like"), list) else []

    policy_instructions = {
        "primary_only": "Mention all conditions in narrative but only include the PRIMARY diagnosis code in the diagnosis list.",
        "partial": "Include the primary diagnosis and a subset of secondary diagnosis codes.",
        "all": "Include all diagnosis codes in the diagnosis list.",
    }[policy]

    ctx = []
    if subject_id is not None:
        ctx.append(f"subject_id: {subject_id}")
    if hadm_id is not None:
        ctx.append(f"hadm_id: {hadm_id}")
    if age is not None:
        ctx.append(f"age_at_admit: {age}")
    if gender:
        ctx.append(f"gender: {gender}")
    if admission_type:
        ctx.append(f"admission_type: {admission_type}")
    if los_days is not None:
        ctx.append(f"length_of_stay_days: {los_days:.1f}")
    if primary_icd10:
        ctx.append(f"primary_icd10: {primary_icd10}")
    if primary_icd10_desc:
        ctx.append(f"primary_icd10_desc: {primary_icd10_desc}")
    if icd10_codes:
        ctx.append(f"icd10_codes: {format_list(icd10_codes, 30)}")
    if icd10_descs:
        ctx.append(f"icd10_descriptions: {format_list(icd10_descs, 10)}")
    if procedures:
        ctx.append(f"procedures_icd10: {format_list(procedures, 20)}")
    if meds:
        ctx.append(f"meds_discharge_like: {format_list(meds, 15)}")

    context_text = "\n".join(ctx)

    system_prompt = (
        "You are a hospital physician writing a discharge summary for a patient being referred to home health services.\n"
        "Write a clinically realistic, concise discharge summary (250-400 words). Maintain authenticity and use appropriate medical terminology."
    )

    user_prompt = f"""
Patient Information and Context:
{context_text}

Instructions:
- Brief history of present illness
- Hospital course
- Key findings and procedures
- Discharge condition and functional status (2-3 mentions relevant to home health)
- Medications and follow-up plan

Coding policy for the DISCHARGE DIAGNOSES section:
{policy_instructions}

Output format:
DISCHARGE SUMMARY
[narrative text]

DISCHARGE DIAGNOSES:
[only include codes based on the policy above]
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def call_model(client, model_name: str, temperature: float, messages, policy: str) -> DischargeSummary:
    return await client.chat.completions.create(
        model=model_name,
        response_model=DischargeSummary,
        messages=messages,
        temperature=temperature,
        extra_body={"metadata": {"policy": policy}},
    )


async def generate_for_row(client, cfg: Config, row: pd.Series, rng: random.Random, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        policy = choose_policy(rng)
        messages = build_messages(row, policy)
        delay = 1.0
        for attempt in range(cfg.max_retries):
            try:
                result: DischargeSummary = await call_model(client, cfg.model_name, cfg.temperature, messages, policy)
                return {
                    "subject_id": int(row["subject_id"]),
                    "hadm_id": int(row["hadm_id"]),
                    "policy": result.policy,
                    "discharge_summary": result.discharge_summary,
                    "diagnosis_codes": result.diagnosis_codes,
                    "model": cfg.model_name,
                }
            except Exception:
                if attempt == cfg.max_retries - 1:
                    raise
                await asyncio.sleep(delay + rng.random())
                delay *= 2


async def run_async(rows: List[pd.Series], cfg: Config) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.concurrency)
    rng = random.Random(17)
    client = instructor.from_openai(AsyncOpenAI())
    tasks = [asyncio.create_task(generate_for_row(client, cfg, r, rng, sem)) for r in rows]
    return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic discharge summaries with OpenAI 4o + Instructor")
    parser.add_argument("--in", dest="in_path", type=str, default="data/processed/structured_datasets.train.parquet")
    parser.add_argument("--out", dest="out_path", type=str, default="data/processed/discharge_summaries.train.parquet")
    parser.add_argument("--model", dest="model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temp", dest="temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=12)
    parser.add_argument("--retries", dest="max_retries", type=int, default=5)
    parser.add_argument("--sample", dest="sample", type=int, default=200)
    args = parser.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in your environment."

    cfg = Config(
        in_path=args.in_path,
        out_path=args.out_path,
        model_name=args.model_name,
        temperature=args.temperature,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        sample=(args.sample if args.sample and args.sample > 0 else None),
    )

    df = pd.read_parquet(cfg.in_path)
    if cfg.sample:
        df = df.sample(n=min(cfg.sample, len(df)), random_state=17)

    rows = [df.iloc[i] for i in range(len(df))]
    results = asyncio.run(run_async(rows, cfg))
    out_df = pd.DataFrame(results)
    out_df.to_parquet(cfg.out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {cfg.out_path}")


if __name__ == "__main__":
    main()
