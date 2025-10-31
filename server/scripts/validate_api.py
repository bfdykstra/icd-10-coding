#!/usr/bin/env python3
"""
Validation script for ICD-10 code checking API.

Loads validation data samples, calls the API with discharge summaries and provided codes,
and compares the API responses to the true missing codes.
"""

import pandas as pd
import numpy as np
import requests
import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict
import time


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# API configuration
API_BASE_URL = "http://localhost:8000"
ENDPOINT = f"{API_BASE_URL}/check-icd-codes/streaming"

# Data paths
VALIDATION_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "structured_dataset_with_discharge_summaries.val.parquet"
ICD_DESCRIPTIONS_PATH = Path(__file__).parent.parent / "data" / "idc10_links" / "icd10_code_descriptions.csv"


def load_icd_descriptions() -> Dict[str, str]:
    """Load ICD code descriptions from CSV file."""
    print(f"Loading ICD code descriptions from {ICD_DESCRIPTIONS_PATH}...")
    try:
        df = pd.read_csv(ICD_DESCRIPTIONS_PATH, usecols=['code_x', 'code_description'])
        # Create a dictionary mapping code to description
        # Handle multiple entries per code by taking the first non-null description
        code_desc_map = {}
        for _, row in df.iterrows():
            code = row['code_x']
            desc = row['code_description']
            if pd.notna(code) and pd.notna(desc) and code not in code_desc_map:
                code_desc_map[code] = str(desc).strip()
        
        print(f"Loaded {len(code_desc_map)} ICD code descriptions")
        return code_desc_map
    except Exception as e:
        print(f"Warning: Could not load ICD descriptions: {e}")
        print("Will use placeholder descriptions")
        return {}


def load_validation_data(num_samples: Optional[int] = None) -> pd.DataFrame:
    """Load validation dataset."""
    print(f"Loading validation data from {VALIDATION_DATA_PATH}...")
    df = pd.read_parquet(VALIDATION_DATA_PATH)
    
    if num_samples:
        df = df.head(num_samples)
    
    print(f"Loaded {len(df)} validation samples")
    return df


def format_existing_codes(
    codes: List[str], 
    code_desc_map: Dict[str, str],
    icd10_codes: Optional[List[str]] = None,
    icd10_descriptions: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """Format ICD codes for API request with descriptions.
    
    Args:
        codes: List of ICD code strings (diagnosis_codes column)
        code_desc_map: Dictionary mapping codes to descriptions (from CSV)
        icd10_codes: Optional list of all ICD codes from dataset
        icd10_descriptions: Optional list of descriptions matching icd10_codes order
    """
    # Create a mapping from icd10_codes to descriptions if available
    code_to_desc = {}
    if icd10_codes and icd10_descriptions:
        for code, desc in zip(icd10_codes, icd10_descriptions):
            if pd.notna(code) and pd.notna(desc):
                code_str = str(code).strip()
                desc_str = str(desc).strip()
                if code_str and desc_str:
                    code_to_desc[code_str] = desc_str
    
    formatted = []
    for code in codes:
        if pd.notna(code) and code:
            code_str = str(code).strip()
            if not code_str:
                continue
            
            # Try to get description from dataset mapping first, then CSV map
            description = code_to_desc.get(code_str)
            if not description:
                description = code_desc_map.get(code_str, f"ICD-10 code {code_str}")
            
            formatted.append({
                "code": code_str,
                "description": description
            })
    return formatted


def parse_sse_response(response: requests.Response) -> Dict:
    """Parse Server-Sent Events (SSE) response from API."""
    result = {
        "missing_codes": [],
        "progress_messages": [],
        "error": None
    }
    
    current_event_type = None
    current_data = None
    
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            # Empty line indicates end of event, process it
            if current_event_type and current_data:
                try:
                    event_data = json.loads(current_data)
                    
                    if current_event_type == 'progress':
                        result["progress_messages"].append(event_data.get('message', ''))
                    
                    elif current_event_type == 'result':
                        result["missing_codes"] = event_data.get('missing_codes', [])
                    
                    elif current_event_type == 'error':
                        result["error"] = event_data.get('message', 'Unknown error')
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing event data: {e}")
                    print(f"   Raw data: {current_data}")
            
            # Reset for next event
            current_event_type = None
            current_data = None
            continue
        
        # Parse SSE format: "event: <type>" or "data: <json>"
        if line.startswith('event:'):
            current_event_type = line.split(':', 1)[1].strip()
        elif line.startswith('data:'):
            current_data = line.split(':', 1)[1].strip()
    
    return result


def call_api(discharge_summary: str, existing_codes: List[Dict[str, str]], timeout: int = 120) -> Dict:
    """Call the API endpoint with discharge summary and existing codes."""
    data = {
        "discharge_summary": discharge_summary,
        "existing_codes": existing_codes
    }
    
    try:
        response = requests.post(ENDPOINT, json=data, stream=True, timeout=timeout)
        response.raise_for_status()
        return parse_sse_response(response)
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to server. Make sure it's running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def compare_results(api_missing_codes: List[Dict], true_missing_codes: List[str]) -> Dict:
    """Compare API results with true missing codes."""
    # Extract code strings from API response
    api_codes = {code['code'] for code in api_missing_codes if 'code' in code}
    
    # Convert true missing codes to set (handle different formats)
    true_codes_set = set()
    for code in true_missing_codes:
        if pd.notna(code) and code:
            true_codes_set.add(str(code))
    
    # Calculate metrics
    true_positives = api_codes & true_codes_set
    false_positives = api_codes - true_codes_set
    false_negatives = true_codes_set - api_codes
    
    precision = len(true_positives) / len(api_codes) if api_codes else 0.0
    recall = len(true_positives) / len(true_codes_set) if true_codes_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_api_codes": len(api_codes),
        "num_true_codes": len(true_codes_set)
    }


def validate_samples(df: pd.DataFrame, code_desc_map: Dict[str, str], num_samples: Optional[int] = None) -> Dict:
    """Validate API on validation samples."""
    if num_samples:
        df = df.head(num_samples)
    
    results = []
    errors = []
    
    print(f"\nValidating {len(df)} samples...")
    print("=" * 80)
    
    for idx, row in df.iterrows():
        print(f"\nSample {idx + 1}/{len(df)}")
        print(f"  hadm_id: {row.get('hadm_id', 'N/A')}")
        
        # Get discharge summary and diagnosis codes
        discharge_summary = row.get('discharge_summary', '')
        if pd.isna(discharge_summary) or not discharge_summary:
            print("  ⚠️  Skipping: No discharge summary")
            continue
        
        diagnosis_codes = row.get('diagnosis_codes', [])
        true_missing_codes = row.get('missing_codes', [])
        icd10_codes = row.get('icd10_codes', [])
        icd10_descriptions = row.get('icd10_descriptions', [])
        # diagnosis_codes is a numpy array, so we need to convert it to a list
        diagnosis_codes = diagnosis_codes.tolist() if isinstance(diagnosis_codes, np.ndarray) else diagnosis_codes
        true_missing_codes = true_missing_codes.tolist() if isinstance(true_missing_codes, np.ndarray) else true_missing_codes
        
        # Handle icd10_codes and icd10_descriptions
        icd10_codes = icd10_codes.tolist() if isinstance(icd10_codes, np.ndarray) else icd10_codes
        icd10_descriptions = icd10_descriptions.tolist() if isinstance(icd10_descriptions, np.ndarray) else icd10_descriptions
        
        # Format existing codes for API
        existing_codes = format_existing_codes(diagnosis_codes, code_desc_map, icd10_codes, icd10_descriptions)
        
        print(f"  Provided codes: {len(existing_codes)}")
        print(f"  True missing codes: {len(true_missing_codes)}")
        
        # Call API
        print("  Calling API...")
        start_time = time.time()
        api_result = call_api(discharge_summary, existing_codes)
        elapsed_time = time.time() - start_time
        
        if api_result.get("error"):
            print(f"  ❌ Error: {api_result['error']}")
            errors.append({
                "idx": idx,
                "hadm_id": row.get('hadm_id', 'N/A'),
                "error": api_result['error']
            })
            continue

        # need to parse the api_result['missing_codes'] to remove the '.' in them
        for code in api_result['missing_codes']:
            code['code'] = code['code'].replace('.', '')
        print('api_result["missing_codes"]:', api_result['missing_codes'])
        print('true_missing_codes:', true_missing_codes)
        
       
        # Compare results
        comparison = compare_results(api_result["missing_codes"], true_missing_codes)
        
        print(f"  ✅ API returned {comparison['num_api_codes']} missing codes")
        print(f"  Precision: {comparison['precision']:.3f}")
        print(f"  Recall: {comparison['recall']:.3f}")
        print(f"  F1: {comparison['f1']:.3f}")
        print(f"  Time: {elapsed_time:.2f}s")
        
        if comparison['false_positives']:
            print(f"  False positives ({len(comparison['false_positives'])}): {comparison['false_positives'][:5]}")
        if comparison['false_negatives']:
            print(f"  False negatives ({len(comparison['false_negatives'])}): {comparison['false_negatives'][:5]}")
        
        results.append({
            "idx": idx,
            "hadm_id": row.get('hadm_id', 'N/A'),
            "num_provided_codes": len(existing_codes),
            "num_true_missing": len(true_missing_codes),
            "num_api_missing": comparison['num_api_codes'],
            "elapsed_time": elapsed_time,
            **comparison
        })
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    return {
        "results": results,
        "errors": errors
    }


def print_summary(validation_results: Dict):
    """Print summary statistics."""
    results = validation_results["results"]
    errors = validation_results["errors"]
    
    if not results:
        print("\n❌ No successful validations")
        return
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal samples processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    # Calculate aggregate metrics
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_time = sum(r['elapsed_time'] for r in results) / len(results)
    
    print(f"\nAverage Metrics:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall: {avg_recall:.3f}")
    print(f"  F1 Score: {avg_f1:.3f}")
    print(f"  Avg Time per Request: {avg_time:.2f}s")
    
    # Count samples with perfect recall
    perfect_recall = sum(1 for r in results if r['recall'] == 1.0)
    print(f"\nSamples with perfect recall: {perfect_recall}/{len(results)} ({perfect_recall/len(results)*100:.1f}%)")
    
    # Count samples with perfect precision
    perfect_precision = sum(1 for r in results if r['precision'] == 1.0)
    print(f"Samples with perfect precision: {perfect_precision}/{len(results)} ({perfect_precision/len(results)*100:.1f}%)")
    
    # Distribution of TP, FP, FN
    total_tp = sum(len(r['true_positives']) for r in results)
    total_fp = sum(len(r['false_positives']) for r in results)
    total_fn = sum(len(r['false_negatives']) for r in results)
    
    print(f"\nTotal Codes:")
    print(f"  True Positives: {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    
    if errors:
        print(f"\n⚠️  Errors encountered:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - Sample {error['idx']} (hadm_id: {error['hadm_id']}): {error['error']}")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ICD-10 code checking API")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to validate (default: all)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})"
    )
    
    args = parser.parse_args()
    
    
    
    # Check if server is running
    print("Checking if API server is running...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        response.raise_for_status()
        print("✅ API server is running")
    except Exception as e:
        print(f"❌ Error connecting to API server: {e}")
        print(f"   Make sure the server is running: python run_server.py")
        sys.exit(1)
    
    # Load data
    code_desc_map = load_icd_descriptions()
    validation_df = load_validation_data(args.num_samples)
    
    # Run validation
    validation_results = validate_samples(validation_df, code_desc_map, args.num_samples)
    
    # Print summary
    print_summary(validation_results)
    
    # Optionally save results to file
    if validation_results["results"]:
        output_file = Path(__file__).parent.parent / "results" / "validation" / "api_validation_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()

