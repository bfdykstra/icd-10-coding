export type PastCase = {
  id: string
  date: string
  clinician: string
  dischargeSummary: string
  existingCodes: Array<{ code: string; description: string }>
  suggestedCodes: Array<{ code: string; description: string; clinicalInfo?: string }>
  acceptedCodes: Array<{ code: string; description: string }>
}

export const mockPastCases: PastCase[] = [
  {
    id: "2024-001",
    date: "2024-01-15",
    clinician: "Dr. Sarah Johnson",
    dischargeSummary: `Patient is a 68-year-old male admitted with acute exacerbation of chronic obstructive pulmonary disease (COPD). Patient presented to the emergency department with increased shortness of breath, productive cough with yellow sputum, and wheezing. 

Medical history includes hypertension, type 2 diabetes mellitus with diabetic neuropathy, and a 40 pack-year smoking history (quit 5 years ago).

During hospitalization, patient was treated with nebulized bronchodilators, systemic corticosteroids, and antibiotics for suspected bacterial superinfection. Chest X-ray showed hyperinflation consistent with COPD without acute infiltrates. Patient's oxygen saturation improved from 88% on room air to 94% on 2L nasal cannula.

Patient was discharged in stable condition with prescriptions for inhaled corticosteroids, long-acting bronchodilators, and a short course of oral prednisone. Follow-up arranged with pulmonology in 2 weeks.`,
    existingCodes: [
      { code: "J44.1", description: "Chronic obstructive pulmonary disease with acute exacerbation" },
      { code: "I10", description: "Essential (primary) hypertension" },
    ],
    suggestedCodes: [
      {
        code: "E11.40",
        description: "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
        clinicalInfo: "Documented in medical history as active condition requiring ongoing management",
      },
      {
        code: "Z87.891",
        description: "Personal history of nicotine dependence",
        clinicalInfo: "40 pack-year smoking history documented, relevant for COPD management",
      },
      {
        code: "J20.9",
        description: "Acute bronchitis, unspecified",
        clinicalInfo: "Productive cough with yellow sputum and antibiotic treatment suggests bacterial bronchitis",
      },
    ],
    acceptedCodes: [
      { code: "J44.1", description: "Chronic obstructive pulmonary disease with acute exacerbation" },
      { code: "I10", description: "Essential (primary) hypertension" },
      { code: "E11.40", description: "Type 2 diabetes mellitus with diabetic neuropathy, unspecified" },
      { code: "Z87.891", description: "Personal history of nicotine dependence" },
      { code: "J20.9", description: "Acute bronchitis, unspecified" },
    ],
  },
  {
    id: "2024-002",
    date: "2024-01-18",
    clinician: "Dr. Michael Chen",
    dischargeSummary: `Patient is a 75-year-old female admitted for management of acute decompensated heart failure. Patient presented with progressive dyspnea on exertion, orthopnea, and bilateral lower extremity edema over the past week.

Past medical history significant for heart failure with reduced ejection fraction (EF 30%), atrial fibrillation on anticoagulation, chronic kidney disease stage 3, and obesity (BMI 34).

Physical examination revealed jugular venous distension, bilateral crackles on lung auscultation, and 3+ pitting edema to the knees. BNP was elevated at 1,850 pg/mL. Chest X-ray showed pulmonary vascular congestion and small bilateral pleural effusions.

Patient was treated with IV diuretics with good response. Daily weights showed 8 kg fluid loss over 4 days. Echocardiogram confirmed reduced EF without significant change from prior. Medications were optimized including uptitration of beta-blocker and ACE inhibitor.

Patient discharged in improved condition with close cardiology follow-up scheduled. Patient and family educated on daily weight monitoring, fluid restriction, and low-sodium diet.`,
    existingCodes: [
      { code: "I50.23", description: "Acute on chronic systolic heart failure" },
      { code: "I48.91", description: "Unspecified atrial fibrillation" },
    ],
    suggestedCodes: [
      {
        code: "N18.3",
        description: "Chronic kidney disease, stage 3",
        clinicalInfo: "Active chronic condition documented in medical history",
      },
      {
        code: "E66.9",
        description: "Obesity, unspecified",
        clinicalInfo: "BMI 34 documented, relevant for heart failure management and prognosis",
      },
      {
        code: "J91.8",
        description: "Pleural effusion in other conditions classified elsewhere",
        clinicalInfo: "Bilateral pleural effusions documented on chest X-ray secondary to heart failure",
      },
      {
        code: "Z79.01",
        description: "Long term (current) use of anticoagulants",
        clinicalInfo: "Patient on anticoagulation for atrial fibrillation",
      },
    ],
    acceptedCodes: [
      { code: "I50.23", description: "Acute on chronic systolic heart failure" },
      { code: "I48.91", description: "Unspecified atrial fibrillation" },
      { code: "N18.3", description: "Chronic kidney disease, stage 3" },
      { code: "E66.9", description: "Obesity, unspecified" },
      { code: "Z79.01", description: "Long term (current) use of anticoagulants" },
    ],
  },
  {
    id: "2024-003",
    date: "2024-01-20",
    clinician: "Dr. Emily Rodriguez",
    dischargeSummary: `Patient is a 82-year-old male admitted after mechanical fall at home resulting in right hip fracture. Patient was found on the floor by family after falling while getting out of bed during the night.

Medical history includes Alzheimer's dementia, benign prostatic hyperplasia, and osteoporosis. Patient lives at home with daughter who provides care.

X-ray confirmed right femoral neck fracture. Patient underwent successful right total hip arthroplasty on hospital day 2. Surgery was uncomplicated. Post-operative course notable for mild delirium managed with reorientation and environmental modifications.

Physical therapy initiated post-operatively with weight-bearing as tolerated. Patient made good progress with mobility using walker. Pain well-controlled with scheduled acetaminophen and PRN oxycodone.

Patient discharged to skilled nursing facility for continued rehabilitation. Medications reconciled and updated. Follow-up with orthopedic surgery scheduled in 2 weeks for wound check and X-ray.`,
    existingCodes: [
      { code: "S72.001A", description: "Fracture of unspecified part of neck of right femur, initial encounter" },
      { code: "W19.XXXA", description: "Unspecified fall, initial encounter" },
    ],
    suggestedCodes: [
      {
        code: "G30.9",
        description: "Alzheimer's disease, unspecified",
        clinicalInfo:
          "Active diagnosis documented in medical history, relevant for fall risk and post-operative delirium",
      },
      {
        code: "M81.0",
        description: "Age-related osteoporosis without current pathological fracture",
        clinicalInfo: "Documented history of osteoporosis, contributing factor to fracture risk",
      },
      {
        code: "N40.0",
        description: "Benign prostatic hyperplasia without lower urinary tract symptoms",
        clinicalInfo: "Active chronic condition requiring ongoing management",
      },
      {
        code: "F05",
        description: "Delirium due to known physiological condition",
        clinicalInfo: "Post-operative delirium documented and managed during hospitalization",
      },
      {
        code: "Z96.641",
        description: "Presence of right artificial hip joint",
        clinicalInfo: "Status post total hip arthroplasty performed during this admission",
      },
    ],
    acceptedCodes: [
      { code: "S72.001A", description: "Fracture of unspecified part of neck of right femur, initial encounter" },
      { code: "W19.XXXA", description: "Unspecified fall, initial encounter" },
      { code: "G30.9", description: "Alzheimer's disease, unspecified" },
      { code: "M81.0", description: "Age-related osteoporosis without current pathological fracture" },
      { code: "F05", description: "Delirium due to known physiological condition" },
      { code: "Z96.641", description: "Presence of right artificial hip joint" },
    ],
  },
]
