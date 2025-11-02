#!/usr/bin/env python3
"""
Validation script for ICD-10 code checking API.

Loads validation data samples, calls the API with discharge summaries and provided codes,
and compares the API responses to the true missing codes.
"""

import pandas as pd
import numpy as np
import aiohttp
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional
import time


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# API configuration
API_BASE_URL = "http://localhost:8000"
ENDPOINT = f"{API_BASE_URL}/check-icd-codes/streaming"

# Data paths
VALIDATION_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "structured_dataset_with_discharge_summaries.val.parquet"
ICD_DESCRIPTIONS_PATH = Path(__file__).parent.parent / "data" / "idc10_links" / "icd10_code_descriptions.csv"
CHECKPOINT_DIR = Path(__file__).parent.parent / "results" / "validation"
CHECKPOINT_FILE = CHECKPOINT_DIR / "validation_checkpoint.json"


def load_icd_descriptions() -> Dict[str, str]:
    """Load ICD code descriptions from CSV file."""
    print(f"Loading ICD code descriptions from {ICD_DESCRIPTIONS_PATH}...")
    try:
        df = pd.read_csv(ICD_DESCRIPTIONS_PATH, usecols=['code_x', 'code_description'])  # type: ignore
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


async def parse_sse_response(response: aiohttp.ClientResponse) -> Dict:
    """Parse Server-Sent Events (SSE) response from API."""
    result = {
        "missing_codes": [],
        "progress_messages": [],
        "error": None
    }
    
    current_event_type = None
    current_data = None
    
    # Buffer for incomplete lines
    buffer = ""
    
    # Read response chunk by chunk and parse line by line
    async for chunk in response.content.iter_chunked(8192):
        buffer += chunk.decode('utf-8')
        
        # Process complete lines
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            line = line.strip()
            
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
    
    # Process any remaining event
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
    
    return result


async def call_api(
    session: aiohttp.ClientSession,
    discharge_summary: str, 
    existing_codes: List[Dict[str, str]], 
    timeout: int = 120
) -> Dict:
    """Call the API endpoint with discharge summary and existing codes."""
    data = {
        "discharge_summary": discharge_summary,
        "existing_codes": existing_codes
    }
    
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with session.post(ENDPOINT, json=data, timeout=timeout_obj) as response:
            response.raise_for_status()
            return await parse_sse_response(response)
    except aiohttp.ClientConnectorError:
        return {"error": "Could not connect to server. Make sure it's running."}
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP {e.status}: {e.message}"
        if e.status == 429 or is_rate_limit_error(error_msg):
            return {"error": f"Rate limit error: {error_msg}", "rate_limited": True}
        return {"error": error_msg}
    except Exception as e:
        error_msg = str(e)
        if is_rate_limit_error(error_msg):
            return {"error": error_msg, "rate_limited": True}
        return {"error": error_msg}


def is_rate_limit_error(error_msg: str) -> bool:
    """Check if error message indicates rate limiting."""
    error_lower = error_msg.lower()
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "quota exceeded",
        "limit exceeded"
    ]
    return any(indicator in error_lower for indicator in rate_limit_indicators)


def save_checkpoint(processed_indices: Set[int], results: List[Dict], errors: List[Dict]) -> None:
    """Save checkpoint to file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "processed_indices": sorted(list(processed_indices)),
        "results": results,
        "errors": errors
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"\n💾 Checkpoint saved: {len(processed_indices)} samples processed")


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint from file if it exists."""
    if not CHECKPOINT_FILE.exists():
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"\n📂 Found checkpoint: {len(checkpoint_data['processed_indices'])} samples already processed")
        return checkpoint_data
    except Exception as e:
        print(f"⚠️  Warning: Could not load checkpoint: {e}")
        return None


def clear_checkpoint() -> None:
    """Clear checkpoint file."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"🗑️  Checkpoint cleared")


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


async def validate_single_sample(
    session: aiohttp.ClientSession,
    idx: int,
    row: pd.Series,
    code_desc_map: Dict[str, str],
    total_samples: int
) -> Optional[Dict]:
    """Validate a single sample."""
    sample_num = idx + 1
    print(f"\nSample {sample_num}/{total_samples}")
    print(f"  hadm_id: {row.get('hadm_id', 'N/A')}")
    
    # Get discharge summary and diagnosis codes
    discharge_summary = row.get('discharge_summary', '')
    if discharge_summary is None or (isinstance(discharge_summary, str) and not discharge_summary) or pd.isna(discharge_summary):
        print("  ⚠️  Skipping: No discharge summary")
        return None
    
    diagnosis_codes = row.get('diagnosis_codes', [])
    true_missing_codes = row.get('missing_codes', [])
    icd10_codes = row.get('icd10_codes', [])
    icd10_descriptions = row.get('icd10_descriptions', [])
    
    # diagnosis_codes is a numpy array, so we need to convert it to a list
    if diagnosis_codes is None:
        diagnosis_codes = []
    else:
        diagnosis_codes = diagnosis_codes.tolist() if isinstance(diagnosis_codes, np.ndarray) else list(diagnosis_codes)
    
    if true_missing_codes is None:
        true_missing_codes = []
    else:
        true_missing_codes = true_missing_codes.tolist() if isinstance(true_missing_codes, np.ndarray) else list(true_missing_codes)
    
    # Handle icd10_codes and icd10_descriptions
    if icd10_codes is None:
        icd10_codes = []
    else:
        icd10_codes = icd10_codes.tolist() if isinstance(icd10_codes, np.ndarray) else list(icd10_codes)
    
    if icd10_descriptions is None:
        icd10_descriptions = []
    else:
        icd10_descriptions = icd10_descriptions.tolist() if isinstance(icd10_descriptions, np.ndarray) else list(icd10_descriptions)
    
    # Format existing codes for API
    existing_codes = format_existing_codes(diagnosis_codes, code_desc_map, icd10_codes, icd10_descriptions)
    
    print(f"  Provided codes: {len(existing_codes)}")
    print(f"  True missing codes: {len(true_missing_codes)}")
    
    # Call API
    print("  Calling API...")
    start_time = time.time()
    api_result = await call_api(session, discharge_summary, existing_codes)
    elapsed_time = time.time() - start_time
    
    if api_result.get("error"):
        print(f"  ❌ Error: {api_result['error']}")
        return {
            "idx": idx,
            "hadm_id": row.get('hadm_id', 'N/A'),
            "error": api_result['error']
        }

    # need to parse the api_result['missing_codes'] to remove the '.' in them
    for code in api_result['missing_codes']:
        code['code'] = code['code'].replace('.', '')        
   
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
    
    return {
        "idx": idx,
        "hadm_id": row.get('hadm_id', 'N/A'),
        "num_provided_codes": len(existing_codes),
        "num_true_missing": len(true_missing_codes),
        "num_api_missing": comparison['num_api_codes'],
        "elapsed_time": elapsed_time,
        **comparison
    }


async def validate_samples(
    df: pd.DataFrame, 
    code_desc_map: Dict[str, str], 
    num_samples: Optional[int] = None,
    batch_size: int = 10,
    checkpoint: Optional[Dict] = None
) -> Dict:
    """Validate API on validation samples using async batching."""
    if num_samples:
        df = df.head(num_samples)
    
    # Initialize results from checkpoint if available
    if checkpoint:
        results = checkpoint.get("results", [])
        errors = checkpoint.get("errors", [])
        processed_indices = set(checkpoint.get("processed_indices", []))
        print(f"Resuming from checkpoint: {len(processed_indices)} samples already processed")
    else:
        results = []
        errors = []
        processed_indices = set()
    
    print(f"\nValidating {len(df)} samples with batch size {batch_size}...")
    print(f"Remaining samples to process: {len(df) - len(processed_indices)}")
    print("=" * 80)
    
    # Create aiohttp session
    connector = aiohttp.TCPConnector(limit=batch_size)
    timeout = aiohttp.ClientTimeout(total=120)
    
    rate_limit_hit = False
    
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process samples in batches
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]
                
                # Skip already processed samples
                batch_indices = [idx for idx in batch_df.index if idx not in processed_indices]
                if not batch_indices:
                    print(f"\nSkipping batch {batch_start // batch_size + 1} (all samples already processed)")
                    continue
                
                batch_df_filtered = df.loc[batch_indices]
                
                print(f"\nProcessing batch {batch_start // batch_size + 1} (samples {batch_start + 1}-{batch_end})...")
                print(f"  Filtered to {len(batch_indices)} unprocessed samples")
                
                # Create tasks for this batch
                tasks = []
                for idx, row in batch_df_filtered.iterrows():
                    task = validate_single_sample(
                        session, idx, row, code_desc_map, len(df)
                    )
                    tasks.append(task)
                
                # Wait for all tasks in batch to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                batch_rate_limited = False
                for result in batch_results:
                    if isinstance(result, Exception):
                        error_msg = str(result)
                        if is_rate_limit_error(error_msg):
                            batch_rate_limited = True
                            rate_limit_hit = True
                        errors.append({
                            "idx": "unknown",
                            "hadm_id": "N/A",
                            "error": error_msg
                        })
                    elif result is None:
                        # Skipped sample
                        continue
                    elif isinstance(result, dict) and "rate_limited" in result:
                        batch_rate_limited = True
                        rate_limit_hit = True
                        errors.append(result)
                    elif isinstance(result, dict) and "error" in result:
                        errors.append(result)
                        if is_rate_limit_error(result.get("error", "")):
                            batch_rate_limited = True
                            rate_limit_hit = True
                    elif isinstance(result, dict):
                        results.append(result)
                        processed_indices.add(result.get("idx"))
                
                # Save checkpoint after each batch
                save_checkpoint(processed_indices, results, errors)
                
                # If rate limited, stop processing
                if batch_rate_limited:
                    print(f"\n⚠️  Rate limit detected! Stopping and saving checkpoint...")
                    break
                
                # Small delay between batches to avoid overwhelming the API
                if batch_end < len(df):
                    await asyncio.sleep(5)
    
    finally:
        # Save final checkpoint
        save_checkpoint(processed_indices, results, errors)
        
        if rate_limit_hit:
            print(f"\n🛑 Stopped due to rate limiting. Progress saved.")
            print(f"   Processed: {len(processed_indices)}/{len(df)} samples")
            print(f"   Run again to resume from checkpoint")
    
    return {
        "results": results,
        "errors": errors,
        "rate_limited": rate_limit_hit
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


async def check_server_running(api_url: str) -> bool:
    """Check if API server is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/", timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                return True
    except Exception:
        return False


async def main_async():
    """Async main validation function."""
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent requests per batch (default: 10)"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint and start fresh"
    )
    
    args = parser.parse_args()
    
    # Update global API_BASE_URL and ENDPOINT if custom URL provide
    
    # Handle checkpoint clearing
    if args.clear_checkpoint:
        clear_checkpoint()
        print("Starting fresh (checkpoint cleared)")

    
    
    
    # Check if server is running
    print("Checking if API server is running...")
    if not await check_server_running(API_BASE_URL):
        print(f"❌ Error connecting to API server")
        print(f"   Make sure the server is running: python run_server.py")
        sys.exit(1)
    print("✅ API server is running")
    
    # Load checkpoint if available
    checkpoint = None if args.clear_checkpoint else load_checkpoint()
    
    # Load data
    code_desc_map = load_icd_descriptions()
    validation_df = load_validation_data(args.num_samples)
    
    # Run validation
    validation_results = await validate_samples(
        validation_df, 
        code_desc_map, 
        args.num_samples,
        batch_size=args.batch_size,
        checkpoint=checkpoint
    )
    
    # Check if we hit rate limit
    rate_limited = validation_results.get("rate_limited", False)
    
    # Print summary
    print_summary(validation_results)
    
    # Optionally save results to file
    if validation_results["results"]:
        output_file = Path(__file__).parent.parent / "results" / "validation" / "api_validation_results_with_rag.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

        # make pandas dataframe from the validation results and get the average recall, precision, and f1 score
        results_df = pd.DataFrame(validation_results["results"])
        avg_recall = results_df['recall'].mean()
        avg_precision = results_df['precision'].mean()
        avg_f1 = results_df['f1'].mean()
        results_df.to_csv(output_file.parent / "api_validation_results_with_rag.csv", index=False)
        print(f"\nAverage Recall: {avg_recall:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average F1 Score: {avg_f1:.3f}")
        
        print(f"\n✅ Results saved to {output_file}")
        
        # Clear checkpoint if we completed successfully
        if not rate_limited:
            clear_checkpoint()
            print("✅ Validation completed successfully! Checkpoint cleared.")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

