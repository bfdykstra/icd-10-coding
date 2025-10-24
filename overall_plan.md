# ICD-10 Code Validation & Enhancement System for Home Health

## Project Plan for Olli Health Application

### Executive Summary

Build an AI-powered ICD-10 code validation and enhancement system that analyzes clinical discharge summaries, validates existing diagnosis codes against narrative documentation, identifies missing codes, and suggests improvements—directly mirroring Olli Health's CodePilot+ workflow. This project demonstrates expertise in clinical NLP, document analysis, healthcare compliance, and full-stack development.

---

## Problem Statement

### The Real Challenge Home Health Agencies Face

When home health agencies receive hospital discharge referrals, they encounter:

- **Incomplete coding:** Discharge summaries often have narrative descriptions of conditions without corresponding ICD-10 codes
- **Incorrect codes:** Provided codes may not align with the narrative documentation or may not be specific enough
- **Missing comorbidities:** Secondary conditions mentioned in text but not coded, leading to lost reimbursement
- **PDGM misalignment:** Codes that don't fit into the 12 Patient-Driven Groupings Model (PDGM) clinical classifications required for home health
- **Documentation gaps:** Insufficient detail to support proper coding and compliance

### What Olli Health Does

Olli Health's certified coders use their proprietary CodePilot+ technology to:

1. Review 100+ pages of referral documents
2. Extract patient history and diagnosis codes
3. Flag inconsistencies and conflicting information
4. Produce revenue-maximizing, compliant diagnosis codes 10x faster than traditional methods

### This Project

Build a similar system that validates existing ICD-10 codes against clinical narratives, suggests additional codes for missed conditions, and identifies documentation gaps—demonstrating the technical foundation for Olli's coding assistance workflow.

---

## Dataset: MIMIC-IV (Structured Data)

### Available Tables

**Core Tables:**

- `diagnoses_icd.csv` - ICD-10 codes assigned (target + validation reference)
- `d_icd_diagnoses.csv` - ICD code descriptions
- `patients.csv` - Demographics
- `admissions.csv` - Admission type, discharge location
- `procedures_icd.csv` - Procedures performed
- `prescriptions.csv` - Medications
- `labevents.csv` - Laboratory results (subset if large)
- `chartevents.csv` - Vital signs (subset if large)

### Data Preparation Strategy

1. **Filter to home health-relevant cases:**

   - Discharge disposition = "HOME WITH HOME IV PROVIDER" or "HOME HEALTH CARE"
   - Primary diagnoses in home health categories (diabetes, heart failure, COPD, wounds, post-surgical care, stroke, hypertension)
   - Patients with 3+ diagnosis codes (to demonstrate multi-code scenarios)

2. **Sample size:** ~1,000-2,000 patient episodes (manageable for synthetic generation)

3. **Create train/validation/test splits:**
   - Training: 70% (for developing validation rules and LLM prompts)
   - Validation: 15% (for tuning)
   - Test: 15% (for final evaluation)

---

## Project Architecture

### Part 1: Synthetic Discharge Summary Generation (30% effort)

#### Why Synthetic Text?

The MIMIC-IV snapshot lacks clinical notes, requiring generation of realistic discharge summaries that mirror real-world referral documents home health agencies receive.

#### Generation Strategy

**Input Data for Each Patient:**

- Demographics (age, gender)
- All assigned ICD-10 codes + descriptions
- Medications list
- Procedures performed
- Key lab values (if available)
- Discharge disposition

**LLM Prompt Template:**

```
You are a hospital physician writing a discharge summary for a patient being referred to home health services.

Patient Information:
- Age: [age], Gender: [gender]
- Admission Date: [date], Discharge Date: [date]
- Discharge Disposition: Home with home health services

Clinical Information:
- Primary Diagnosis: [ICD code description]
- Secondary Diagnoses: [list of conditions]
- Procedures Performed: [procedures]
- Medications at Discharge: [medication list]
- Relevant Labs: [key values]

Instructions:
1. Write a realistic discharge summary (250-400 words) including:
   - Brief history of present illness
   - Hospital course
   - Key findings and procedures
   - Discharge condition and functional status
   - Medications and follow-up plan

2. IMPORTANT CODING VARIATIONS TO SIMULATE REAL-WORLD SCENARIOS:
   - For 50% of patients: Mention ALL conditions in narrative but ONLY include primary diagnosis code
   - For 30% of patients: Include primary + some (but not all) secondary diagnosis codes
   - For 20% of patients: Include all diagnosis codes
   - Randomly use clinical terminology that implies conditions without stating them directly
   - Include 2-3 functional status mentions relevant to home health (mobility, ADLs, wound care needs)

3. Use realistic medical terminology and maintain clinical authenticity.

Output format:
DISCHARGE SUMMARY
[narrative text]

DISCHARGE DIAGNOSES:
[only include codes based on the percentage allocation above]
```

**Quality Controls:**

1. **Sample review:** Manually review 50 generated summaries for clinical plausibility
2. **Consistency checks:** Verify medications align with diagnoses
3. **Variation validation:** Ensure proper distribution of coding completeness scenarios
4. **Entity verification:** Use NER to confirm conditions mentioned in text match source data
5. **Documentation:** Save generation parameters and random seeds for reproducibility

**Generation Approach:**

- Use GPT-4, Claude, or Llama 3.1/3.2 (8B model sufficient)
- Batch generation with rate limiting
- Store generated summaries with metadata linking back to MIMIC patient IDs
- Cost estimate: ~$20-50 for 1,500 summaries with GPT-4

#### Synthetic Data Output

For each patient episode:

- `patient_id` - MIMIC patient identifier
- `discharge_summary` - Generated narrative text
- `provided_codes` - ICD codes included in discharge (simulating incomplete coding)
- `true_codes` - All actual ICD codes from MIMIC (ground truth)
- `missing_codes` - Codes mentioned in narrative but not provided
- `metadata` - Demographics, medications, procedures

### Part 2: Clinical NLP & Code Extraction (25% effort)

#### Core NLP Pipeline

**1. Clinical Named Entity Recognition (NER)**

- **Tool:** use a model from https://huggingface.co/OpenMed/models (medical NER models)
- **Extract:**
  - Diseases and conditions
  - Symptoms
  - Medications
  - Procedures
  - Anatomical locations
  - Functional limitations (mobility, ADLs, wounds)

**2. Clinical Text Embeddings**

- **Model:** Bio_ClinicalBERT or PubMedBERT https://github.com/lindvalllab/BioClinical-ModernBERT, or https://huggingface.co/Simonlee711/Clinical_ModernBERT
- **Purpose:**
  - Semantic similarity between narrative mentions and ICD descriptions
  - Find best-matching ICD codes for extracted conditions

**3. Medication-to-Condition Mapping**

- Create lookup table: medication → common associated diagnoses
- Example: Metformin → Diabetes (E11.x), Lisinopril → Hypertension (I10)
- Use to validate narrative-code consistency

**4. Clinical Context Analysis**

- Identify severity indicators (acute, chronic, exacerbation, stable)
- Extract temporal information (on admission, during stay, at discharge)
- Flag negation ("no evidence of", "denies history of")

#### Code Suggestion Engine

**Input:** Extracted clinical entities from narrative

**Process:**

1. **Map entities to ICD-10 codes:**

   - Exact matching on `d_icd_diagnoses` descriptions
   - Fuzzy matching for partial matches
   - Semantic search using embeddings (entity ↔ ICD description similarity)

2. **Rank candidate codes by:**

   - Confidence score (NER confidence × semantic similarity)
   - Specificity level (prefer specific over general codes)
   - PDGM classification (prioritize home health-relevant codes)
   - Supporting evidence strength (multiple mentions, medication support)

3. **Apply clinical rules:**
   - Combination code requirements (e.g., acute + chronic heart failure)
   - Required secondary codes (manifestation codes, laterality)
   - PDGM-specific requirements

### Part 3: Code Validation & Enhancement Logic (25% effort)

#### Validation Components

**1. Code-to-Narrative Consistency Check**

- For each provided ICD code:
  - Is there narrative evidence supporting this diagnosis?
  - Text span extraction showing supporting evidence
  - Confidence score (strong/moderate/weak/unsupported)

**2. Missing Code Detection**

- Identify conditions mentioned in narrative but not coded:
  - Clinical entities extracted but no matching ICD code provided
  - Medications implying conditions
  - Procedures suggesting underlying diagnoses

**3. Code Specificity Analysis**

- Check if more specific codes are available:
  - E11.9 (Type 2 diabetes NOS) → E11.65 (Type 2 diabetes with hyperglycemia) if labs support
  - I50.9 (Heart failure NOS) → I50.23 (Acute on chronic systolic heart failure) if narrative specifies

**4. PDGM Classification Validation**

- Verify primary diagnosis fits one of 12 PDGM clinical groups:
  - MMTA - MS & Major Neurological
  - MMTA - Neuro/Rehab
  - MMTA - Wounds
  - Behavioral Health
  - Complex Nursing Interventions
  - Cardiac & Circulatory
  - Respiratory
  - Infectious Disease
  - Endocrine
  - GI/GU
  - Surgical Aftercare
  - Musculoskeletal
- Flag "unacceptable diagnoses" that don't fit PDGM

**5. Comorbidity Scoring**

- Identify low vs. high comorbidity adjustments (14 subgroups)
- Suggest additional codes that would qualify for comorbidity adjustment
- Calculate potential reimbursement impact

**6. Documentation Gap Identification**

- Required elements missing from narrative:
  - Functional status (required for OASIS)
  - Wound staging details
  - Severity descriptors needed for specific codes
  - Lab values supporting diagnostic codes

#### Output Structure

For each patient, generate validation report:

```python
{
  "patient_id": "12345",
  "provided_codes": ["E11.9", "I50.9"],
  "validation_results": {
    "E11.9": {
      "status": "VALID",
      "evidence": "Patient has type 2 diabetes...",
      "specificity_suggestion": "E11.65",
      "reasoning": "Narrative mentions hyperglycemia, consider more specific code"
    },
    "I50.9": {
      "status": "VALID",
      "evidence": "Admitted for heart failure exacerbation...",
      "specificity_suggestion": "I50.23",
      "reasoning": "Narrative indicates acute on chronic systolic CHF"
    }
  },
  "missing_codes": [
    {
      "suggested_code": "I10",
      "description": "Essential hypertension",
      "evidence": "Patient on lisinopril; narrative mentions 'hypertensive'",
      "confidence": 0.92,
      "impact": "Adds comorbidity adjustment"
    }
  ],
  "pdgm_analysis": {
    "primary_diagnosis": "E11.9",
    "clinical_group": "Endocrine",
    "valid_for_pdgm": true,
    "comorbidity_adjustment": "Low",
    "suggested_comorbidity_codes": ["I10"]
  },
  "documentation_gaps": [
    "No functional status mentioned - required for OASIS M1800-M1860",
    "Diabetic complications not specified - could support E11.65 vs E11.9"
  ],
  "reimbursement_impact": {
    "current_estimated_reimbursement": "$2,800",
    "with_suggested_codes": "$3,100",
    "improvement": "+$300 (10.7%)"
  }
}
```

### Part 4: Interactive Dashboard (20% effort)

#### Backend (FastAPI + Python)

**Endpoints:**

- `POST /api/analyze` - Submit discharge summary + provided codes
- `GET /api/validate/{patient_id}` - Retrieve validation results
- `POST /api/generate-summary` - Demo endpoint for synthetic generation
- `GET /api/icd-lookup/{code}` - ICD code details and PDGM classification

**Services:**

- `NLPService` - Clinical entity extraction
- `CodeValidator` - Validation logic
- `PDGMClassifier` - PDGM grouping and scoring
- `SyntheticGenerator` - Discharge summary generation (optional demo)

#### Frontend (React + TypeScript)

**Main Views:**

**1. Upload & Input View**

- Text area for discharge summary
- Input fields for provided ICD codes
- Patient demographics (optional)
- "Analyze" button

**2. Validation Dashboard**
Split into sections:

**A. Code Validation Results**

- Table showing each provided code:
  - ✅ Valid / ⚠️ Needs specificity / ❌ Unsupported
  - Supporting evidence (highlighted text spans)
  - Specificity suggestions with reasoning
  - PDGM classification

**B. Missing Codes Detection**

- Table of suggested additional codes:
  - Code + description
  - Confidence score
  - Evidence from narrative
  - Impact on reimbursement/comorbidity
  - "Add to codes" button

**C. Documentation Analysis**

- PDGM Classification summary
- Comorbidity adjustment status
- Reimbursement estimate (current vs. optimized)

**D. Documentation Gaps**

- List of missing information with severity
- Suggestions for improvement
- OASIS-specific requirements flagged

**E. Narrative View**

- Discharge summary with highlights:
  - Green: Conditions that are coded
  - Yellow: Conditions mentioned but not coded
  - Red: Codes without narrative support
- Side panel showing extracted entities

**3. Comparison View**

- Before/After code sets
- Impact summary
- Export report (PDF)

#### UI Features

- Collapsible sections for clean interface
- Evidence hover tooltips (show full text context)
- Color-coded severity indicators
- Export validation report as PDF
- "Demo Mode" with pre-loaded examples

### Part 5: Evaluation & Validation (Ongoing)

#### Metrics

**1. Code Suggestion Accuracy**

- Precision: Of suggested codes, how many are in ground truth?
- Recall: Of true codes, how many were suggested?
- F1 Score for missing code detection

**2. Validation Accuracy**

- True positive rate: Correctly identifying valid codes
- False positive rate: Incorrectly flagging valid codes
- Agreement with ground truth on code appropriateness

**3. Entity Extraction Performance**

- NER precision/recall on medical conditions
- Matching accuracy for condition → ICD code mapping

**4. Clinical Utility Metrics**

- Time to review (simulated)
- Documentation gap identification rate
- PDGM classification accuracy

#### Error Analysis

- **False negatives:** Missed conditions that should be coded
- **False positives:** Suggested codes not appropriate
- **Specificity errors:** Suggesting wrong level of specificity
- **Clinical reasoning failures:** Missing context or nuance

---

## Project Timeline (10 Days)

### Days 1-2: Data Preparation & Exploration

**Tasks:**

- [ ] Load MIMIC-IV tables and explore structure
- [ ] Filter to home health-relevant patients (~1,500 patients)
- [ ] Analyze ICD code distributions by PDGM groups
- [ ] Create patient cohorts with varying code completeness
- [ ] Set up train/val/test splits (70/15/15)
- [ ] Document data quality and characteristics

**Deliverables:**

- Filtered dataset with metadata
- Data quality report
- Patient cohort statistics

### Day 3: Synthetic Discharge Summary Generation

**Tasks:**

- [ ] Design LLM prompt template with coding variation logic
- [ ] Set up generation pipeline (batch processing with rate limits)
- [ ] Generate discharge summaries for training set (~1,000 patients)
- [ ] Manual quality review of 50 samples
- [ ] Validate coding variation distribution (50/30/20 split)
- [ ] Store summaries with metadata linking to MIMIC IDs

**Deliverables:**

- 1,000+ synthetic discharge summaries
- Generation script and prompts
- Quality review report

### Days 4-5: NLP Pipeline Development

**Tasks:**

- [ ] Set up spaCy with clinical NER models
- [ ] Implement entity extraction pipeline
- [ ] Build medication-to-condition lookup table
- [ ] Implement Bio_ClinicalBERT embedding generation
- [ ] Create ICD code semantic search functionality
- [ ] Build entity → ICD code mapping with confidence scoring
- [ ] Test on sample summaries and tune parameters

**Deliverables:**

- Functional NLP pipeline
- Entity extraction accuracy metrics
- Code suggestion engine

### Day 6: Validation Logic Implementation

**Tasks:**

- [ ] Implement code-to-narrative consistency checker
- [ ] Build missing code detection algorithm
- [ ] Create code specificity analyzer
- [ ] Implement PDGM classification logic
- [ ] Build comorbidity scoring system
- [ ] Create documentation gap identifier
- [ ] Test validation on training set

**Deliverables:**

- Complete validation engine
- Validation accuracy metrics
- Rule-based logic documentation

### Day 7: PDGM & Reimbursement Analysis

**Tasks:**

- [ ] Map all MIMIC ICD codes to PDGM groups
- [ ] Implement comorbidity adjustment logic (14 subgroups)
- [ ] Build reimbursement impact estimator
- [ ] Create PDGM validation reports
- [ ] Test on validation set and refine

**Deliverables:**

- PDGM classification system
- Reimbursement calculator
- Home health-specific validation rules

### Days 8-9: Dashboard Development

**Tasks:**

- [ ] Set up FastAPI backend with endpoints
- [ ] Integrate NLP and validation services
- [ ] Build React frontend with TypeScript
- [ ] Implement main validation dashboard views
- [ ] Create evidence highlighting UI
- [ ] Add export functionality (PDF reports)
- [ ] Test end-to-end workflow
- [ ] Deploy locally or to cloud (Railway/Render)

**Deliverables:**

- Functional web application
- API documentation
- Demo-ready interface

### Day 10: Testing, Documentation & Polish

**Tasks:**

- [ ] Run evaluation on test set (15% held out)
- [ ] Generate evaluation metrics and charts
- [ ] Perform error analysis on failures
- [ ] Write comprehensive README
- [ ] Create demo video (2-3 minutes)
- [ ] Prepare presentation materials
- [ ] Document methodology and design decisions
- [ ] Create "Relevance to Olli Health" document

**Deliverables:**

- Test set results and error analysis
- Complete project documentation
- Demo video
- GitHub repository ready to share

---

## Technical Stack

### Data Processing & ML

- **Python 3.10+**
- **pandas, numpy** - data manipulation
- **scikit-learn** - evaluation metrics, similarity
- **scipy** - statistical analysis

### NLP & Clinical Models

- **spaCy** - clinical NER (en_core_sci_lg, en_ner_bc5cdr_md)
- **transformers (Hugging Face)** - Bio_ClinicalBERT, PubMedBERT
- **sentence-transformers** - efficient embeddings
- **fuzzywuzzy / rapidfuzz** - fuzzy matching for ICD descriptions
- **OpenAI API / Anthropic API / Llama** - synthetic summary generation
- **Instructor** - structuring LLM outputs

### Validation & Rules

- **Custom rule engine** - PDGM logic, comorbidity scoring
- **JSON schemas** - validation report structure

### Visualization & Reporting

- **matplotlib, seaborn** - evaluation charts
- **plotly** - interactive dashboard visualizations
- **reportlab / weasyprint** - PDF export

### Full Stack

- **FastAPI** - backend REST API
- **React + TypeScript** - frontend
- **Material-UI or Tailwind CSS** - UI components
- **Axios** - API calls
- **React Highlight Words** - text evidence highlighting

### Development & Deployment

- **Jupyter notebooks** - exploration and prototyping
- **Git** - version control
- **pytest** - testing
- **Docker** - containerization (optional)
- **Railway / Render / Vercel** - deployment options

---

## Expected Outcomes

### Deliverables

**1. Synthetic Dataset**

- 1,000+ discharge summaries with varying code completeness
- Linked to MIMIC-IV ground truth codes
- Metadata and generation documentation

**2. NLP & Validation System**

- Clinical entity extraction pipeline
- ICD code suggestion engine
- Multi-component validation system
- PDGM classification and scoring

**3. Interactive Dashboard**

- Web application for code validation
- Evidence-based highlighting
- Missing code suggestions
- Documentation gap identification
- Reimbursement impact calculator

**4. Evaluation Report**

- Performance metrics on held-out test set
- Error analysis and failure modes
- Comparison of validation approaches
- Clinical utility assessment

**5. Documentation**

- Comprehensive README with setup instructions
- Methodology documentation
- API reference
- Demo video (2-3 minutes)
- "Relevance to Olli Health" document

### Success Metrics

**Code Suggestion Performance:**

- **Precision:** >0.75 for missing code detection
- **Recall:** >0.70 for identifying codes in narrative
- **F1 Score:** >0.72 overall

**Validation Accuracy:**

- **Code validity classification:** >0.85 accuracy
- **PDGM classification:** >0.95 accuracy (well-defined rules)
- **Specificity suggestions:** >0.70 agreement with clinical logic

**System Performance:**

- **Analysis time:** <5 seconds per patient
- **Dashboard load time:** <2 seconds
- **End-to-end workflow:** <30 seconds from upload to results

**Clinical Utility:**

- Identifies 2+ missing codes per patient on average
- Flags 1-2 documentation gaps per patient
- Estimates 5-15% potential reimbursement improvement

---

## Relevance to Olli Health

### Direct Alignment with Olli's CodePilot+ Workflow

**What Olli Does:**

1. ✅ Ingests referral documents and clinical notes
2. ✅ Extracts patient history and diagnosis codes
3. ✅ Flags inconsistencies and conflicts
4. ✅ Certified coders review and finalize codes
5. ✅ Produces revenue-maximizing, compliant claims

**What This Project Demonstrates:**

1. ✅ Document analysis and information extraction
2. ✅ Code validation against narrative evidence
3. ✅ Inconsistency and gap detection
4. ✅ Augmenting human review with AI assistance
5. ✅ Revenue optimization through complete coding

### Job Requirements Met

**From the Job Listing:**

- ✅ **NLP expertise:** Clinical NER, embeddings, semantic search
- ✅ **Healthcare datasets:** ICD-10, PDGM, OASIS considerations
- ✅ **Data pipelines:** Feature extraction, structured outputs
- ✅ **Full stack:** React + FastAPI with complete UI
- ✅ **Flexible problem solving:** Combining rule-based + ML approaches
- ✅ **HIPAA considerations:** Documented security principles
- ✅ **Startup mentality:** Built end-to-end in 10 days

### Home Health Domain Expertise

**Demonstrates Understanding of:**

- PDGM clinical groupings and payment model
- Comorbidity adjustments for reimbursement
- OASIS documentation requirements
- Face-to-face encounter coding rules
- Incomplete referral challenges
- Revenue optimization within compliance

### Key Talking Points for Application

**1. Problem Understanding:**
_"I built an ICD-10 validation system that addresses the core challenge home health agencies face: referral documents with incomplete or missing diagnosis codes. The system validates provided codes against narrative evidence, suggests missing codes for conditions mentioned in the discharge summary, and identifies documentation gaps—directly mirroring Olli's CodePilot+ workflow."_

**2. Technical Approach:**
_"Using clinical NLP (spaCy + Bio_ClinicalBERT), I extract medical entities from discharge summaries, map them to ICD-10 codes through semantic search, and apply PDGM-specific validation rules. The system achieved 75% precision in identifying missing codes while maintaining clinical accuracy through evidence-based suggestions."_

**3. Home Health Focus:**
_"The validation logic specifically addresses home health coding requirements: PDGM clinical groupings, comorbidity adjustments, and OASIS documentation needs. The system estimates reimbursement impact and flags codes that don't fit PDGM classifications—ensuring both compliance and revenue optimization."_

**4. Data Strategy:**
_"Since MIMIC-IV lacked clinical notes, I generated 1,000+ synthetic discharge summaries using LLMs with carefully designed prompts that simulate real-world scenarios—50% with only primary diagnosis coded, 30% partially coded, 20% fully coded—matching the incomplete referral challenges Olli's coders face daily."_

**5. Full Stack Implementation:**
_"Built an interactive dashboard (React + FastAPI) that highlights evidence for each code, visualizes missing diagnoses, and generates exportable validation reports. The interface supports the human-in-the-loop workflow where AI augments expert review rather than replacing it."_

---

## Future Enhancements

### After Initial Completion

**1. Multi-Document Support**

- Ingest face-to-face encounter notes + discharge summary
- Reconcile information across documents
- Flag conflicts between sources

**2. Real Clinical Notes**

- Obtain MIMIC-IV note data through PhysioNet
- Compare synthetic vs. real note performance
- Refine NLP models on actual clinical text

**3. OASIS Assessment Integration**

- Map ICD codes to OASIS item requirements
- Generate OASIS documentation checklists
- Suggest functional status coding based on narrative

**4. Active Learning**

- Identify low-confidence predictions for human review
- Incorporate feedback to improve suggestions
- Build user-specific preference models

**5. Medication Interaction Analysis**

- Flag medication combinations suggesting unreported conditions
- Identify potential drug interactions
- Suggest medication reconciliation needs

**6. Historical Trend Analysis**

- Track patient across multiple episodes
- Identify chronic vs. acute conditions
- Suggest longitudinal coding consistency

**7. Compliance Audit Mode**

- Retrospective review of coded charts
- Identify systematic coding patterns
- Generate compliance reports for agencies

**8. Production Readiness**

- API rate limiting and authentication
- Batch processing for multiple patients
- Model versioning and A/B testing
- Logging and monitoring
- HIPAA compliance hardening

---

## Repository Structure

```
home-health-code-validator/
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_generation.ipynb
│   ├── 03_nlp_pipeline.ipynb
│   ├── 04_validation_logic.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── load_mimic.py
│   │   ├── synthetic_generator.py
│   │   └── preprocessing.py
│   ├── nlp/
│   │   ├── entity_extractor.py
│   │   ├── embeddings.py
│   │   ├── code_mapper.py
│   │   └── clinical_ner.py
│   ├── validation/
│   │   ├── code_validator.py
│   │   ├── pdgm_classifier.py
│   │   ├── comorbidity_scorer.py
│   │   └── documentation_checker.py
│   ├── models/
│   │   └── validation_models.py
│   └── utils/
│       ├── icd_lookup.py
│       ├── medication_mapping.py
│       └── config.py
├── dashboard/
│   ├── backend/
│   │   ├── main.py (FastAPI)
│   │   ├── api/
│   │   │   ├── analyze.py
│   │   │   ├── validate.py
│   │   │   └── lookup.py
│   │   ├── services/
│   │   │   ├── nlp_service.py
│   │   │   ├── validation_service.py
│   │   │   └── pdgm_service.py
│   │   └── models/
│   │       └── schemas.py
│   └── frontend/
│       ├── src/
│       │   ├── components/
│       │   │   ├── ValidationDashboard.tsx
│       │   │   ├── CodeTable.tsx
│       │   │   ├── NarrativeView.tsx
│       │   │   ├── MissingCodes.tsx
│       │   │   └── DocumentationGaps.tsx
│       │   ├── services/
│       │   │   └── api.ts
│       │   ├── types/
│       │   │   └── validation.ts
│       │   ├── App.tsx
│       │   └── index.tsx
│       ├── package.json
│       └── tsconfig.json
├── data/
│   ├── raw/ (MIMIC-IV files)
│   ├── processed/
│   │   ├── filtered_patients.csv
│   │   └── train_val_test_splits/
│   └── synthetic/
│       ├── discharge_summaries.json
│       └── generation_metadata.json
├── results/
│   ├── evaluation/
│   │   ├── metrics.json
│   │   ├── error_analysis.md
│   │   └── figures/
│   ├── examples/
│   │   └── sample_validation_reports/
│   └── demo_video.mp4
├── docs/
│   ├── methodology.md
│   ├── olli_health_relevance.md
│   ├── api_reference.md
│   └── pdgm_reference.md
├── tests/
│   ├── test_nlp.py
│   ├── test_validation.py
│   └── test_api.py
└── scripts/
    ├── generate_summaries.py
    ├── run_validation.py
    └── deploy.sh
```

---

## Risk Mitigation

### Potential Challenges & Solutions

**Challenge 1: Synthetic summaries not realistic enough**

- **Mitigation:** Manual review + iterative prompt refinement
- **Validation:** Compare entity distributions with real clinical notes from literature
- **Backup:** Download smaller public clinical note datasets for comparison

**Challenge 2: Low NLP extraction accuracy**

- **Mitigation:** Use established clinical NER models (spaCy sci models)
- **Validation:** Test on known clinical datasets first (i2b2, n2c2)
- **Backup:** Combine rule-based extraction with ML for higher precision

**Challenge 3: PDGM logic complexity**

- **Mitigation:** Focus on most common clinical groups (80% of cases)
- **Validation:** Reference official CMS PDGM grouper logic
- **Backup:** Implement basic classification first, add complexity iteratively

**Challenge 4: Dashboard development time**

- **Mitigation:** Use UI component libraries (Material-UI) for rapid development
- **Validation:** Start with simple views, add features iteratively
- **Backup:** Focus on backend API first, basic frontend sufficient for demo

**Challenge 5: Evaluation without ground truth clinical reasoning**

- **Mitigation:** Use MIMIC codes as ground truth, focus on objective metrics
- **Validation:** Error analysis on disagreements to understand system limits
- **Backup:** Document assumptions and limitations clearly

---

## Questions for Refinement

Before starting, consider:

1. **LLM Choice:** Which LLM will you use for synthetic generation? (GPT-4o-mini for cost, GPT-4 for quality, or Llama 3.2 for free/local)
2. **Sample Size:** Start with 500 or go for full 1,500 summaries?
3. **Dashboard Deployment:** Local only or deploy to cloud for live demo?
4. **Evaluation Focus:** Emphasize technical metrics or clinical utility?
5. **Time Allocation:** Spend more time on NLP accuracy vs. dashboard polish?
6. **Documentation Depth:** Basic README or comprehensive methodology guide?

---

## Success Metrics for This Project

### Technical Success

- ✅ System validates 1,000+ discharge summaries in <10 minutes
- ✅ Achieves >75% precision/recall on missing code detection
- ✅ PDGM classification accuracy >95%
- ✅ Dashboard loads in <2 seconds
- ✅ Clean, documented, deployable codebase

### Application Success

- ✅ Demonstrates deep understanding of Olli's workflow
- ✅ Shows relevant technical skills (NLP, full-stack, healthcare data)
- ✅ Proves ability to ship complete projects quickly
- ✅ Highlights home health domain knowledge
- ✅ Professional presentation (GitHub, demo, documentation)

### Learning Success

- ✅ Understand PDGM payment model and coding requirements
- ✅ Experience with clinical NLP and healthcare datasets
- ✅ Practice with LLM-based data augmentation
- ✅ Portfolio piece for future healthcare tech opportunities
- ✅ Networking opportunity in home health space

---

## Additional Resources

### Healthcare Coding References

- **CMS PDGM Overview:** https://www.cms.gov/medicare/payment/prospective-payment-systems/home-health
- **ICD-10-CM Guidelines:** https://www.cms.gov/medicare/coding-billing/icd-10-codes
- **OASIS Manual:** https://www.cms.gov/medicare/quality/home-health-quality/oasis-overview
- **Home Health Coding Guidelines:** Various professional coding associations

### Clinical NLP Resources

- **spaCy Medical Models:** https://github.com/allenai/scispacy
- **Bio_ClinicalBERT:** https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- **OpenMed NER Models:** https://huggingface.co/OpenMed/models?p=1 (specialized medical entity recognition models)
- **n2c2 Challenges:** https://n2c2.dbmi.hms.harvard.edu/
- **Clinical NLP Papers:** PubMed search for recent methods

### MIMIC-IV Resources

- **PhysioNet:** https://physionet.org/content/mimiciv/
- **MIMIC Code Repository:** https://github.com/MIT-LCP/mimic-code
- **Documentation:** https://mimic.mit.edu/docs/iv/

### Similar Projects for Inspiration

- **Clinical coding systems** on GitHub
- **Healthcare NLP papers** from ACL/EMNLP clinical tracks
- **ICD coding research** from AMIA and medical informatics conferences

---

## Appendix: Home Health Coding Primer

### PDGM 101 (Patient-Driven Groupings Model)

**What is PDGM?**
Medicare's payment model for home health (effective Jan 2020) that determines reimbursement based on patient characteristics including:

1. **Clinical grouping** (primary diagnosis → 1 of 12 groups)
2. **Functional impairment** (low/medium/high)
3. **Comorbidity adjustment** (none/low/high)
4. **Timing** (early vs. late in episode)
5. **Referral source** (community vs. institutional)

**12 Clinical Groups:**

1. MMTA - MS & Major Neurological Disorders
2. MMTA - Neuro/Rehab
3. MMTA - Wounds
4. Behavioral Health
5. Complex Nursing Interventions
6. Cardiac & Circulatory
7. Respiratory
8. Infectious Disease
9. Endocrine
10. GI/GU
11. Surgical Aftercare
12. Musculoskeletal

**Why It Matters:**

- Primary diagnosis must map to a PDGM group or claim is denied
- Only ~43,000 of 70,000+ ICD codes are PDGM-eligible
- Secondary diagnoses affect comorbidity adjustments (±5-10% payment)
- Complete, accurate coding = appropriate reimbursement

### Common Home Health Conditions

**Top Diagnoses:**

- Diabetes (E11.x) - Endocrine group
- Heart Failure (I50.x) - Cardiac group
- COPD (J44.x) - Respiratory group
- Pressure ulcers (L89.x) - Wounds group
- Hip/knee replacement aftercare (Z96.x) - Surgical aftercare
- Stroke sequelae (I69.x) - Neuro/Rehab
- Hypertension (I10) - Often secondary for comorbidity

**Comorbidity Examples:**
Secondary diagnoses that increase payment:

- Diabetes with complications
- Multiple chronic conditions
- Mental health conditions
- Complex medication regimens
- History of falls

### OASIS Connection

**OASIS (Outcome and Assessment Information Set):**

- Required assessment for Medicare home health patients
- Includes ICD-10 codes in items M1021-M1028
- Codes must align with physician face-to-face documentation
- Used for quality measures and payment determination

**Why Coding Matters for OASIS:**

- Incomplete codes → delayed OASIS submission → payment delays
- Incorrect codes → quality measure errors → star rating impact
- Missing comorbidities → inadequate care planning
- Documentation gaps → compliance issues

---

## Final Checklist Before Submission

### Code Quality

- [ ] All code follows PEP 8 style guidelines
- [ ] Functions have clear docstrings
- [ ] No hardcoded paths or credentials
- [ ] Requirements.txt is complete and tested
- [ ] Git history shows incremental progress

### Documentation

- [ ] README has clear setup instructions
- [ ] Architecture diagram included
- [ ] API endpoints documented
- [ ] Example usage provided
- [ ] Limitations acknowledged

### Demo Preparation

- [ ] Dashboard deployed or video recorded
- [ ] Example validation reports prepared
- [ ] Key results summarized (1-page PDF)
- [ ] Screenshots of main features
- [ ] 2-3 minute demo video edited

### Application Materials

- [ ] Cover letter tailored to Olli Health
- [ ] Resume highlights relevant experience
- [ ] GitHub repository public and polished
- [ ] LinkedIn updated with project
- [ ] Email draft ready to send

### Technical Validation

- [ ] Code runs on fresh environment
- [ ] Tests pass (if implemented)
- [ ] Dashboard loads without errors
- [ ] Example workflows complete successfully
- [ ] Performance metrics documented

---
