// import validation_results from "@/assets/pastCases/validation_results.json";
import validation_results from "@/assets/pastCases/validation_results_gpt-4.1-mini_new_prompt.json";

export type PastCase = {
  id: string;
  date: string;
  clinician: string;
  dischargeSummary: string;

  existingCodes: Array<string>;
  suggestedCodes: Array<string>;
  acceptedCodes: Array<string>;
  rejectedCodes: Array<string>;
};

export const pastCases: PastCase[] = validation_results.map((result) => ({
  id: result.hadm_id.toString(),
  date: new Date(result.date).toLocaleDateString(),
  clinician: result.clinician,
  dischargeSummary: result.discharge_summary,
  existingCodes: result.diagnosis_codes,
  suggestedCodes: result.suggested_codes,
  acceptedCodes: result.accepted_codes,
  rejectedCodes: result.rejected_codes,
}));
