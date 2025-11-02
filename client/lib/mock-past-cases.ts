import validation_results from "@/assets/pastCases/validation_results.json";

export type PastCase = {
  id: string;
  date: Date;
  clinician: string;
  dischargeSummary: string;

  existingCodes: Array<string>;
  suggestedCodes: Array<string>;
  acceptedCodes: Array<string>;
  rejectedCodes: Array<string>;
};
// make a random date between 2024-01-01 and 2024-12-31
const randomDate = () => new Date(2024, 0, 1 + Math.random() * 365);

const randomClinician = () => {
  const clinicians = [
    "Dr. Sarah Johnson",
    "Dr. Michael Chen",
    "Dr. Emily Rodriguez",
  ];
  return clinicians[Math.floor(Math.random() * clinicians.length)];
};

export const pastCases: PastCase[] = validation_results.map((result) => ({
  id: result.hadm_id.toString(),
  date: randomDate(),
  clinician: randomClinician(),
  dischargeSummary: result.discharge_summary,
  existingCodes: result.diagnosis_codes,
  suggestedCodes: result.suggested_codes,
  acceptedCodes: result.accepted_codes,
  rejectedCodes: result.rejected_codes,
}));
