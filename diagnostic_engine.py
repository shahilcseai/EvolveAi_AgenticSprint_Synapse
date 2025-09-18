import os
import json
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
import streamlit as st 

class DiagnosticEngine:
    def __init__(self):
        # --- CORRECTED LINE ---
        # This line now correctly retrieves your API key by its NAME from the environment variables.
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable in your .env file.")
            # Stop the app from continuing if the key is missing.
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Medical diagnostic system prompt
        self.system_prompt = """
        You are an advanced AI medical diagnostic assistant designed to analyze patient data and provide evidence-based diagnostic insights. Your role is to serve as a clinical decision support tool for healthcare professionals.

        CORE PRINCIPLES:
        - Provide differential diagnoses with confidence scores (0.0 to 1.0)
        - Prioritize patient safety with red flag condition detection
        - Base recommendations on current medical literature and guidelines
        - Include ICD-10 codes when appropriate
        - Consider risk stratification (High, Medium, Low)
        - Provide clinical reasoning for each diagnosis

        RESPONSE FORMAT:
        Return a JSON object with the following structure:
        {
            "red_flags": [
                {
                    "condition": "Emergency condition name",
                    "reasoning": "Clinical reasoning for urgency",
                    "action": "Immediate action required",
                    "urgency": "HIGH/MEDIUM"
                }
            ],
            "diagnoses": [
                {
                    "condition": "Medical condition name",
                    "confidence_score": 0.0-1.0,
                    "risk_level": "High/Medium/Low",
                    "specialty": "Medical specialty",
                    "icd_10_code": "ICD-10 code if applicable",
                    "clinical_reasoning": "Detailed clinical reasoning",
                    "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
                    "next_steps": "Recommended next steps for evaluation/treatment"
                }
            ],
            "validation": {
                "overall_confidence": 0.0-1.0,
                "evidence_strength": "Strong/Moderate/Weak",
                "recommendation_level": "A/B/C/D",
                "limitations": "Analysis limitations and considerations"
            },
            "reasoning": "Overall AI diagnostic reasoning process"
        }

        SAFETY GUIDELINES:
        - Always emphasize that AI analysis requires clinical validation
        - Highlight any critical or time-sensitive conditions
        - Consider patient demographics, vital signs, and clinical context
        - Flag incomplete or insufficient data for proper diagnosis
        - Recommend appropriate healthcare provider consultation
        """

    def analyze_patient_data(self, patient_data: List[Dict], confidence_threshold: float = 0.3, 
                            max_diagnoses: int = 8, enable_red_flags: bool = True) -> Dict[str, Any]:
        """
        Analyze patient data and return diagnostic insights
        """
        try:
            # Prepare patient data summary
            data_summary = self._prepare_data_summary(patient_data)
            
            # --- DEBUGGING LINE ADDED HERE ---
            print("--- DEBUG: Data Summary Sent to AI ---")
            print(json.dumps(data_summary, indent=2))
            print("-------------------------------------\n")
            # --- END DEBUGGING LINE ---

            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the following patient data and provide diagnostic insights:

            PATIENT DATA:
            {json.dumps(data_summary, indent=2)}

            ANALYSIS PARAMETERS:
            - Minimum confidence threshold: {confidence_threshold}
            - Maximum diagnoses: {max_diagnoses}
            - Red flag detection: {'Enabled' if enable_red_flags else 'Disabled'}

            Please provide a comprehensive diagnostic analysis following the specified JSON format.
            Focus on evidence-based medicine and prioritize patient safety.
            """

            # Generate AI response
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,  # Low temperature for consistent medical analysis
                top_p=0.8,
                max_output_tokens=4096
            )
            
            full_prompt = self.system_prompt + "\n\n" + analysis_prompt
            response = self.model.generate_content(full_prompt, generation_config=generation_config)

            # --- DEBUGGING LINE ADDED HERE ---
            print("--- RAW API RESPONSE TEXT ---")
            print(response.text)
            print("-----------------------------\n")
            # --- END DEBUGGING LINE ---

            if response.text:
                # Parse JSON response
                diagnostic_results = json.loads(response.text)
                
                # Filter diagnoses by confidence threshold
                if diagnostic_results.get('diagnoses'):
                    filtered_diagnoses = [
                        d for d in diagnostic_results['diagnoses'] 
                        if d.get('confidence_score', 0) >= confidence_threshold
                    ]
                    # Limit to max_diagnoses
                    diagnostic_results['diagnoses'] = filtered_diagnoses[:max_diagnoses]
                
                # Remove red flags if disabled
                if not enable_red_flags:
                    diagnostic_results['red_flags'] = []
                
                return diagnostic_results
            else:
                raise ValueError("Empty response from AI model")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse AI response: {e}")
            return self._get_error_response("Failed to parse AI diagnostic response")
        
        except Exception as e:
            logging.error(f"Diagnostic analysis error: {e}")
            return self._get_error_response(f"Diagnostic analysis failed: {str(e)}")

    def _prepare_data_summary(self, patient_data: List[Dict]) -> Dict[str, Any]:
        """
        Prepare and structure patient data for AI analysis
        """
        summary = {
            "patient_count": len(patient_data),
            "demographics": {},
            "vital_signs": {},
            "symptoms": [],
            "medical_history": [],
            "medications": [],
            "laboratory_results": {},
            "imaging_results": [],
            "clinical_notes": []
        }

        for record in patient_data:
            # Extract demographics
            for key, value in record.items():
                key_lower = key.lower()
                
                if any(demo in key_lower for demo in ['age', 'gender', 'sex', 'race', 'ethnicity']):
                    summary["demographics"][key] = value
                
                # Extract vital signs
                elif any(vital in key_lower for vital in ['temperature', 'temp', 'blood_pressure', 'bp', 
                                                         'heart_rate', 'hr', 'respiratory_rate', 'rr', 
                                                         'oxygen', 'o2', 'pulse', 'weight', 'height', 'bmi']):
                    summary["vital_signs"][key] = value
                
                # Extract symptoms and complaints
                elif any(symptom in key_lower for symptom in ['symptom', 'complaint', 'chief_complaint',
                                                             'present', 'pain', 'ache', 'discomfort']):
                    if isinstance(value, str) and value.strip():
                        summary["symptoms"].append(f"{key}: {value}")
                    elif isinstance(value, list):
                        summary["symptoms"].extend([f"{key}: {v}" for v in value if v])
                
                # Extract medical history
                elif any(hist in key_lower for hist in ['history', 'medical_history', 'past_medical',
                                                       'surgical', 'family_history', 'allergy', 'allergies']):
                    if isinstance(value, str) and value.strip():
                        summary["medical_history"].append(f"{key}: {value}")
                    elif isinstance(value, list):
                        summary["medical_history"].extend([f"{key}: {v}" for v in value if v])
                
                # Extract medications
                elif any(med in key_lower for med in ['medication', 'drug', 'prescription', 'treatment']):
                    if isinstance(value, str) and value.strip():
                        summary["medications"].append(f"{key}: {value}")
                    elif isinstance(value, list):
                        summary["medications"].extend([f"{key}: {v}" for v in value if v])
                
                # Extract laboratory results
                elif any(lab in key_lower for lab in ['lab', 'blood', 'urine', 'glucose', 'cholesterol',
                                                     'hemoglobin', 'hematocrit', 'wbc', 'rbc', 'platelet']):
                    summary["laboratory_results"][key] = value
                
                # Extract imaging results
                elif any(img in key in key_lower for img in ['xray', 'x-ray', 'ct', 'mri', 'ultrasound', 'imaging']):
                    if isinstance(value, str) and value.strip():
                        summary["imaging_results"].append(f"{key}: {value}")
                
                # Extract clinical notes
                elif any(note in key_lower for note in ['note', 'assessment', 'plan', 'impression', 'observation']):
                    if isinstance(value, str) and value.strip():
                        summary["clinical_notes"].append(f"{key}: {value}")

        return summary

    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Return standardized error response
        """
        return {
            "red_flags": [],
            "diagnoses": [],
            "validation": {
                "overall_confidence": 0.0,
                "evidence_strength": "Insufficient",
                "recommendation_level": "N/A",
                "limitations": f"Analysis failed: {error_message}"
            },
            "reasoning": f"Unable to complete diagnostic analysis due to: {error_message}. Please consult with a healthcare professional for proper medical evaluation."
        }

    def validate_diagnostic_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate the structure and content of diagnostic results
        """
        required_keys = ["red_flags", "diagnoses", "validation", "reasoning"]
        
        if not all(key in results for key in required_keys):
            return False
        
        # Validate diagnoses structure
        if results.get("diagnoses"):
            for diagnosis in results["diagnoses"]:
                required_diagnosis_keys = ["condition", "confidence_score", "clinical_reasoning"]
                if not all(key in diagnosis for key in required_diagnosis_keys):
                    return False
                
                # Validate confidence score range
                confidence = diagnosis.get("confidence_score", 0)
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    return False
        
        return True

    def get_specialty_recommendations(self, diagnoses: List[Dict]) -> Dict[str, List[str]]:
        """
        Group diagnoses by medical specialty for referral recommendations
        """
        specialty_groups = {}
        
        for diagnosis in diagnoses:
            specialty = diagnosis.get('specialty', 'General Medicine')
            condition = diagnosis.get('condition', 'Unknown')
            
            if specialty not in specialty_groups:
                specialty_groups[specialty] = []
            
            specialty_groups[specialty].append(condition)
        
        return specialty_groups
