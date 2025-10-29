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

    def semantic_search_strategy(self, query_text, max_results=100):
        """
        Perform plain semantic search using discharge summary.
        """
        if not query_text or pd.isna(query_text):
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=max_results
            )
            
            return results['ids'][0] if results['ids'] else []
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def keyword_search_strategy(self, entities, query_text="medical condition", max_results=100):
        """
        Use ChromaDB keyword search with $contains operator.
        """
        if not entities:
            return []
        
        # Build OR query for all entities
        where_conditions = [{"$contains": entity.lower()} for entity in entities if entity]
        
        if not where_conditions:
            return []
        
        try:
            # Query with keyword filtering
            results = self.collection.query(
                query_texts=[query_text],
                where_document={"$or": where_conditions} if len(where_conditions) > 1 else where_conditions[0],
                n_results=max_results
            )
            
            return results['ids'][0] if results['ids'] else []
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []


    def triple_hybrid_strategy(
      self,
      discharge_summary: Optional[str],
      entities: List[str],
      icd_codes_list=None,            # kept for signature compatibility; unused
      icd_documents_list=None,        # kept for signature compatibility; unused
      icd_metadata_list=None,         # kept for signature compatibility; unused
      max_results: int = 100,
  ):
      all_candidates: Dict[str, float] = {}

      entities_lower = []
      seen = set()
      for e in (entities or []):
          e = (e or "").lower().strip()
          if e and e not in seen:
              seen.add(e)
              entities_lower.append(e)

      entity_batch_limit = 32
      n_per_query = min(50, max(10, max_results))

      # Batched entity queries (tag-ish signal)
      if entities_lower:
          try:
              res = self.collection.query(
                  query_texts=entities_lower[:entity_batch_limit],
                  n_results=n_per_query,
                  include=["distances"],
              )
              for i, ids in enumerate(res.get("ids", []) or []):
                  dists = res.get("distances", [[]])[i] if res.get("distances") else []
                  N = max(1, len(ids))
                  for j, code in enumerate(ids):
                      sim = 1.0 - dists[j] if dists and j < len(dists) and dists[j] is not None else (N - j) / N
                      all_candidates[code] = all_candidates.get(code, 0.0) + 3.0 * float(sim)
          except Exception:
              pass

      # Keyword OR on entities
      try:
          where_conditions = [{"$contains": e} for e in entities_lower[:entity_batch_limit]]
          query_text = (discharge_summary or "condition")[:500]
          where_document = (
              {"$or": where_conditions} if len(where_conditions) > 1 else (where_conditions[0] if where_conditions else None)
          )
          res_kw = self.collection.query(
              query_texts=[query_text],
              where_document=where_document,
              n_results=n_per_query,
              include=["distances"],
          )
          ids_kw = res_kw.get("ids", [[]])[0] if res_kw.get("ids") else []
          d_kw = res_kw.get("distances", [[]])[0] if res_kw.get("distances") else []
          N = max(1, len(ids_kw))
          for j, code in enumerate(ids_kw):
              sim = 1.0 - d_kw[j] if d_kw and j < len(d_kw) and d_kw[j] is not None else (N - j) / N
              all_candidates[code] = all_candidates.get(code, 0.0) + 2.0 * float(sim)
      except Exception:
          pass

      # Semantic on summary
      try:
          if discharge_summary:
              res_sem = self.collection.query(
                  query_texts=[discharge_summary[:2000]],
                  n_results=n_per_query,
                  include=["distances"],
              )
              ids_sem = res_sem.get("ids", [[]])[0] if res_sem.get("ids") else []
              d_sem = res_sem.get("distances", [[]])[0] if res_sem.get("distances") else []
              N = max(1, len(ids_sem))
              for j, code in enumerate(ids_sem):
                  sim = 1.0 - d_sem[j] if d_sem and j < len(d_sem) and d_sem[j] is not None else (N - j) / N
                  all_candidates[code] = all_candidates.get(code, 0.0) + 1.0 * float(sim)
      except Exception:
          pass

      sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
      return [code for code, _ in sorted_candidates[:max_results]]

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
      # Dedup while preserving order
      seen = set()
      entities = []
      for e in disease_entities + drug_entities + anatomy_entities:
          if e and e not in seen:
              seen.add(e)
              entities.append(e)

      combined_scores: Dict[str, float] = {}
      # Bound cost
      entity_batch_limit = 32
      n_per_query = min(50, max(10, 5 * k))

      # 1) Batched "tag-based" approximation: embed entities together once
      if entities:
          q_entities = entities[:entity_batch_limit]
          try:
              res = self.collection.query(
                  query_texts=q_entities,
                  n_results=n_per_query,
                  include=["distances"],
              )
              ids_list = res.get("ids", [])
              dists_list = res.get("distances", [])
              for i, ids in enumerate(ids_list or []):
                  dists = dists_list[i] if dists_list and i < len(dists_list) else []
                  N = max(1, len(ids))
                  for j, code in enumerate(ids):
                      # Distance -> similarity; fallback to rank if missing
                      sim = 1.0 - dists[j] if dists and j < len(dists) and dists[j] is not None else (N - j) / N
                      combined_scores[code] = combined_scores.get(code, 0.0) + weights[0] * float(sim)
          except Exception:
              pass

      # 2) Keyword search with OR contains on entities + head of summary
      try:
          kw_entities = entities[:entity_batch_limit]
          where_conditions = [{"$contains": e} for e in kw_entities] if kw_entities else []
          query_text = (discharge_summary or "condition")[:500]
          where_document = (
              {"$or": where_conditions}
              if len(where_conditions) > 1
              else (where_conditions[0] if where_conditions else None)
          )

          res_kw = self.collection.query(
              query_texts=[query_text],
              where_document=where_document,
              n_results=n_per_query,
              include=["distances"],
          )
          ids_kw = res_kw.get("ids", [[]])[0] if res_kw.get("ids") else []
          d_kw = res_kw.get("distances", [[]])[0] if res_kw.get("distances") else []
          N = max(1, len(ids_kw))
          for j, code in enumerate(ids_kw):
              sim = 1.0 - d_kw[j] if d_kw and j < len(d_kw) and d_kw[j] is not None else (N - j) / N
              combined_scores[code] = combined_scores.get(code, 0.0) + weights[1] * float(sim)
      except Exception:
          pass

      # 3) Semantic search on full summary (truncate for embedding cost)
      try:
          if discharge_summary:
              summary_q = discharge_summary[:2000]
              res_sem = self.collection.query(
                  query_texts=[summary_q],
                  n_results=n_per_query,
                  include=["distances"],
              )
              ids_sem = res_sem.get("ids", [[]])[0] if res_sem.get("ids") else []
              d_sem = res_sem.get("distances", [[]])[0] if res_sem.get("distances") else []
              N = max(1, len(ids_sem))
              for j, code in enumerate(ids_sem):
                  sim = 1.0 - d_sem[j] if d_sem and j < len(d_sem) and d_sem[j] is not None else (N - j) / N
                  combined_scores[code] = combined_scores.get(code, 0.0) + weights[2] * float(sim)
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


