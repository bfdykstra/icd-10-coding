import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import chromadb


def _normalize_code(icd_code: str) -> str:
    if pd.isna(icd_code):
        return ""
    return re.sub(r"[^A-Za-z0-9.]", "", str(icd_code)).upper()


def _to_text(value: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _flatten_candidates(items: Iterable[str]) -> List[str]:
    values: List[str] = []
    for v in items:
        if not isinstance(v, str):
            continue
        values.append(v)
    return values


def _tokenize_entities(entities_field: Optional[str]) -> List[str]:
    if not isinstance(entities_field, str) or not entities_field:
        return []
    # Entities are typically serialized like: "['Hypertension' 'Diabetes']" or JSON-like arrays
    # We normalize by stripping quotes/brackets and splitting on commas or spaces between items.
    text = entities_field.strip()
    # Replace various quotes and separators
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    # Split on comma first; if no comma, split on double-spaces between items
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in text.split("  ") if p.strip()]
    # Fallback: split on single space if still one piece and contains separators
    if len(parts) <= 1 and "  " not in text and "," not in text:
        parts = [p.strip() for p in re.split(r"\s{2,}|\s\|\s", text) if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in text.split(" ") if p.strip()]
    return parts


@dataclass
class CandidateSuggestion:
    code: str
    description: str
    score: float
    components: Dict[str, float]
    matched_keywords: List[str]


class CandidateGenerator:
    def __init__(
        self,
        chroma_path: str = "/Users/benjamindykstra/development/icd-10-coding/icd10_embeddings",
        collection_name: str = "icd10_embeddings",
        model_name: str = "lokeshch19/ModernPubMedBERT",
    ) -> None:
        # Initialize Chroma client/collection matching your notebook setup
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=False))
        self.collection = self.client.get_collection(collection_name, embedding_function=self.embedding_fn)

    def _build_keyword_query(self, discharge_summary: str, disease_entities: List[str], drug_entities: List[str], anatomy_entities: List[str]) -> str:
        parts: List[str] = []
        if disease_entities:
            parts.extend(_flatten_candidates(disease_entities))
        if drug_entities:
            parts.extend(_flatten_candidates(drug_entities))
        if anatomy_entities:
            parts.extend(_flatten_candidates(anatomy_entities))
        # include first 500 chars of summary like the notebook
        head = (discharge_summary or "")[:500]
        if head:
            parts.append(head)
        return ", ".join([p for p in parts if p])

    def _accumulate_scores(self, all_scores: Dict[str, float], ids: List[str], base_weight: float) -> None:
        if not ids:
            return
        n = max(1, len(ids))
        for rank, code in enumerate(ids):
            score = base_weight * (n - rank) / n
            all_scores[code] = all_scores.get(code, 0.0) + score

    def suggest_candidates(
        self,
        discharge_summary: str,
        already_coded: Iterable[str],
        disease_entities_field: Optional[str],
        drug_entities_field: Optional[str],
        anatomy_entities_field: Optional[str],
        k: int = 10,
        weights: Tuple[float, float, float] = (3.0, 2.0, 1.0),  # tag, keyword, semantic
    ) -> List[CandidateSuggestion]:
        # Inputs
        disease_entities = [e.lower() for e in _tokenize_entities(disease_entities_field)]
        drug_entities = [e.lower() for e in _tokenize_entities(drug_entities_field)]
        anatomy_entities = [e.lower() for e in _tokenize_entities(anatomy_entities_field)]
        entities = [e for e in (disease_entities + drug_entities + anatomy_entities) if e]

        # Strategy 11: combine three sources with weights
        combined_scores: Dict[str, float] = {}

        # 1) Tag-based: treat each entity as a high-weight query (approximation of metadata tag hits)
        for ent in entities:
            try:
                res = self.collection.query(query_texts=[ent], n_results=100)
                ids = res.get("ids", [[]])[0] if res and res.get("ids") else []
                self._accumulate_scores(combined_scores, ids, base_weight=weights[0])
            except Exception:
                continue

        # 2) Keyword search: combine entities + head of summary into one query
        try:
            kw_query = self._build_keyword_query(discharge_summary, disease_entities, drug_entities, anatomy_entities)
            if kw_query:
                res_kw = self.collection.query(query_texts=[kw_query], n_results=100)
                ids_kw = res_kw.get("ids", [[]])[0] if res_kw and res_kw.get("ids") else []
                self._accumulate_scores(combined_scores, ids_kw, base_weight=weights[1])
        except Exception:
            pass

        # 3) Semantic search: full discharge summary
        try:
            if discharge_summary:
                res_sem = self.collection.query(query_texts=[discharge_summary], n_results=100)
                ids_sem = res_sem.get("ids", [[]])[0] if res_sem and res_sem.get("ids") else []
                self._accumulate_scores(combined_scores, ids_sem, base_weight=weights[2])
        except Exception:
            pass

        # Rank and filter out already-coded
        already = {_normalize_code(c) for c in already_coded if isinstance(c, str)}
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_codes = [c for c, _ in ranked if _normalize_code(c) not in already]

        # Fetch descriptions from collection metadatas/documents
        top_codes = ranked_codes[:k]
        suggestions: List[CandidateSuggestion] = []
        if top_codes:
            try:
                got = self.collection.get(ids=top_codes, include=["metadatas", "documents"])
                meta_map: Dict[str, Dict[str, str]] = {}
                doc_map: Dict[str, Optional[str]] = {}
                for i, cid in enumerate(got.get("ids", [])):
                    md = None
                    if got.get("metadatas") and i < len(got["metadatas"]):
                        md = got["metadatas"][i]
                    meta_map[cid] = md or {}
                    doc_val = None
                    if got.get("documents") and i < len(got["documents"]):
                        doc_val = got["documents"][i]
                    doc_map[cid] = doc_val
            except Exception:
                meta_map, doc_map = {}, {}

            for code in top_codes:
                score = combined_scores.get(code, 0.0)
                md = meta_map.get(code, {})
                desc = md.get("code_description") or md.get("title") or (doc_map.get(code) or "")
                suggestions.append(
                    CandidateSuggestion(
                        code=code,
                        description=str(desc) if desc else "",
                        score=float(score),
                        components={
                            "tag": float(weights[0]),
                            "keyword": float(weights[1]),
                            "semantic": float(weights[2]),
                        },
                        matched_keywords=[],
                    )
                )

        return suggestions


def _parse_codes_field(codes_field: Optional[str]) -> List[str]:
    if not isinstance(codes_field, str):
        return []
    # Expected formats like: "['I5032' 'E118' 'I10']" or JSON-like lists
    # Replace brackets/quotes and split on spaces/commas
    text = codes_field.strip()
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    parts = re.split(r"[\s,]+", text)
    return [p for p in (_normalize_code(p) for p in parts) if p]


def preview_from_dataset(
    generator: CandidateGenerator,
    dataset_csv_path: str,
    limit: int = 3,
) -> List[Dict[str, object]]:
    df = pd.read_csv(dataset_csv_path)
    rows = []
    for _, row in df.head(limit).iterrows():
        discharge_summary = str(row.get("discharge_summary", ""))
        already_coded = _parse_codes_field(row.get("diagnosis_codes", None))
        suggestions = generator.suggest_candidates(
            discharge_summary=discharge_summary,
            already_coded=already_coded,
            disease_entities_field=row.get("disease_entities", None),
            drug_entities_field=row.get("drug_entities", None),
            anatomy_entities_field=row.get("anatomy_entities", None),
            k=10,
        )
        rows.append(
            {
                "hadm_id": int(row.get("hadm_id", -1)) if pd.notna(row.get("hadm_id", np.nan)) else None,
                "primary_icd10": row.get("primary_icd10", None),
                "already_coded": already_coded,
                "suggestions": [s.__dict__ for s in suggestions],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate ICD-10 codes using triple-hybrid retrieval via ChromaDB.")
    parser.add_argument("--chroma_path", default="/Users/benjamindykstra/development/icd-10-coding/icd10_embeddings")
    parser.add_argument("--collection", default="icd10_embeddings")
    parser.add_argument("--model_name", default="lokeshch19/ModernPubMedBERT")
    parser.add_argument("--dataset_csv", default="/Users/benjamindykstra/development/icd-10-coding/data/processed/structured_dataset_with_discharge_summaries_ner_features.train.csv")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", default="/Users/benjamindykstra/development/icd-10-coding/data/processed/missing_code_candidates.sample.jsonl")
    args = parser.parse_args()

    gen = CandidateGenerator(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        model_name=args.model_name,
    )
    preview = preview_from_dataset(gen, args.dataset_csv, limit=args.limit)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for row in preview:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(preview)} rows to {args.output}")


if __name__ == "__main__":
    main()


