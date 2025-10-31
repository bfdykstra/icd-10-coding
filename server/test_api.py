#!/usr/bin/env python3
"""
Simple test script for the ICD-10 code checker API.
Run the server first with: python run_server.py
"""

import requests
import json

# Example discharge summary
DISCHARGE_SUMMARY = """
DISCHARGE SUMMARY

Patient Name: [REDACTED]
Date of Admission: [DATE]
Date of Discharge: [DATE]

HISTORY OF PRESENT ILLNESS:
The patient is a 68-year-old male with a history of type 2 diabetes mellitus, 
hypertension, and chronic kidney disease who presented to the emergency department 
with complaints of progressive dyspnea and lower extremity edema over the past 
two weeks. He reports orthopnea and paroxysmal nocturnal dyspnea.

HOSPITAL COURSE:
The patient was admitted to the cardiology service. Physical examination revealed 
bilateral lower extremity pitting edema, jugular venous distension, and crackles 
at bilateral lung bases. Chest X-ray showed pulmonary congestion. Echocardiogram 
revealed an ejection fraction of 35% with diastolic dysfunction. Labs showed 
elevated BNP at 850 pg/mL, creatinine 2.1 mg/dL (baseline 1.8), and HbA1c of 8.2%.

The patient was treated with IV diuretics with good response. His home medications 
were optimized including initiation of sacubitril-valsartan. Blood glucose was 
managed with insulin.

DISCHARGE CONDITION:
The patient is being discharged in stable condition with improved dyspnea and 
reduced edema. He is ambulatory with assistance and requires home health services 
for medication management and monitoring.

DISCHARGE MEDICATIONS:
1. Sacubitril-valsartan 24/26 mg twice daily
2. Furosemide 40 mg daily
3. Metformin 1000 mg twice daily
4. Insulin glargine 20 units at bedtime
5. Amlodipine 10 mg daily
6. Atorvastatin 40 mg daily

FOLLOW-UP:
Cardiology clinic in 2 weeks
Primary care in 1 week
"""

# Existing codes (incomplete list)
EXISTING_CODES = [
    {"code": "I10", "description": "Essential (primary) hypertension"},
    {"code": "E119", "description": "Type 2 diabetes mellitus without complications"}
]

def test_streaming_endpoint():
    """Test the streaming endpoint"""
    url = "http://localhost:8000/check-icd-codes/streaming"
    
    data = {
        "discharge_summary": DISCHARGE_SUMMARY,
        "existing_codes": EXISTING_CODES
    }
    
    print("Sending request to:", url)
    print("\nExisting codes:")
    for code in EXISTING_CODES:
        print(f"  - {code['code']}: {code['description']}")
    print("\n" + "="*80)
    print("Streaming response:\n")
    
    try:
        response = requests.post(url, json=data, stream=True, timeout=60)
        response.raise_for_status()
        
        current_event_type = None
        current_data = None
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                # Empty line indicates end of event, process it
                if current_event_type and current_data:
                    try:
                        event_data = json.loads(current_data)
                        
                        if current_event_type == 'progress':
                            print(f"📊 Progress: {event_data.get('message', '')}")
                        
                        elif current_event_type == 'result':
                            print("\n✅ Results received:")
                            missing_codes = event_data.get('missing_codes', [])
                            
                            if not missing_codes:
                                print("  No missing codes identified.")
                            else:
                                print(f"\n  Found {len(missing_codes)} potentially missing code(s):\n")
                                for i, code in enumerate(missing_codes, 1):
                                    print(f"  {i}. Code: {code['code']}")
                                    print(f"     Description: {code['description']}")
                                    print(f"     Confidence: {code['confidence'].upper()}")
                                    clinical_info = code.get('clinicalInfo', code.get('clinical_evidence', 'N/A'))
                                    print(f"     Clinical Evidence: {clinical_info[:200] if clinical_info else 'N/A'}...")
                                    print()
                        
                        elif current_event_type == 'done':
                            print("✅ Processing complete!")
                        
                        elif current_event_type == 'error':
                            print(f"❌ Error: {event_data.get('message', 'Unknown error')}")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Error parsing event data: {e}")
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
        
        print("="*80)
        print("✅ Test completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server.")
        print("   Make sure the server is running: python run_server.py")
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out.")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise e

def test_health_check():
    """Test the health check endpoint"""
    url = "http://localhost:8000/"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print("✅ Health check passed:", response.json())
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("ICD-10 Code Checker API Test")
    print("="*80)
    print()
    
    # First check if server is running
    if test_health_check():
        print()
        test_streaming_endpoint()
    else:
        print("\n⚠️  Server is not running. Start it with: python run_server.py")

