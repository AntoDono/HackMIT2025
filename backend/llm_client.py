import os
import json
import re
from typing import Dict, Any
import pandas as pd
import numpy as np
from cerebras.cloud.sdk import Cerebras
from groq import Groq

class LLMClient:
    """Client for LLM API integration supporting both Cerebras and Groq."""
    
    def __init__(self):
        self.provider = os.getenv('PROVIDER', 'cerebras').lower()
        
        if self.provider == 'groq':
            self.api_key = os.getenv('GROQ_API_KEY')
            self.model = "openai/gpt-oss-120b"
            self.client = Groq(api_key=self.api_key) if self.api_key else None
        else:  # Default to cerebras
            self.api_key = os.getenv('CEREBRAS_API_KEY')
            self.model = "gpt-oss-120b"
            self.client = Cerebras(api_key=self.api_key) if self.api_key else None
    
    def call_llm(self, prompt: str) -> str:
        """Make actual API call to the configured LLM provider."""
        if not self.client:
            provider_key = f"{self.provider.upper()}_API_KEY"
            raise ValueError(f"No {provider_key} found - API key is required")
        
        try:
            if self.provider == 'groq':
                completion_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1,
                    max_completion_tokens=8192,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=False,
                    stop=None
                )
            else:  # cerebras
                completion_response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model,
                    stream=False,
                    max_completion_tokens=65536,
                    temperature=1,
                    top_p=1,
                    reasoning_effort="medium"
                )
            
            # Extract content from response
            if completion_response.choices and len(completion_response.choices) > 0:
                return completion_response.choices[0].message.content
            else:
                raise ValueError(f"No response content from {self.provider.title()} API")
                
        except Exception as e:
            print(f"Error calling {self.provider.title()} API: {e}")
            raise e
    
    def generate_song_description(self, classified_emotion: str, eeg_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a therapeutic song description using LLM to bring user's emotion back to baseline.
        
        Args:
            classified_emotion: The classified emotional state of the user
            eeg_analysis: Dictionary containing EEG analysis results from eeg_stats.py
            
        Returns:
            Dict[str, str]: Dictionary containing 'song_reasoning' and 'song_description'
        """
        try:
            # Create the prompt for the LLM
            prompt = self._build_therapeutic_prompt(classified_emotion, eeg_analysis)
            
            # Make actual LLM API call
            llm_response = self.call_llm(prompt)
            
            # Parse JSON response
            song_data = self._parse_json_response(llm_response)
            
            return song_data
            
        except Exception as e:
            print(f"Error generating song description: {e}")
            # Fallback description
            return {
                'song_reasoning': "The user needs calming music to restore emotional balance and reduce stress levels.",
                'song_description': "calm ambient music at 75 BPM with soft piano and strings, peaceful and relaxing atmosphere designed to restore emotional balance"
            }
    
    def _build_therapeutic_prompt(self, classified_emotion: str, eeg_analysis: Dict[str, Any]) -> str:
        """Build the prompt for the LLM to generate therapeutic music descriptions."""
        # Convert EEG analysis to JSON-serializable format
        serializable_analysis = self._make_json_serializable(eeg_analysis)
        
        prompt = f"""You are a music therapist AI that creates therapeutic music descriptions to help users return to emotional baseline (neutral state).

Current User State:
- Classified Emotion: {classified_emotion}
- EEG Analysis Data: {json.dumps(serializable_analysis, indent=2)}

Your task is to analyze this emotional and neurological data and create a specific music description that will therapeutically guide the user back to a calm, neutral emotional state. Consider:

1. The current emotional state and what musical elements would counterbalance it
2. The brainwave patterns and what they suggest about the user's mental state
3. Specific musical parameters (BPM, instruments, genre, mood) that promote emotional balance
4. Keep the description concise and to the point.
5. Song reasoning should be a short, concise explanation of the user's emotional state and the music's purpose and techniques to bring them back to emotional baseline.

Example:

```json
{{
  "song_reasoning": "The user is feeling stressed and anxious. The music should be calming and soothing to help them relax and feel better.",
  "song_description": "90 BPM, soft piano, warm strings, and subtle nature sounds, bird noises, nature."
}}
```


Respond with a JSON object containing both song_reasoning and song_description fields:

{{
  "song_reasoning": "Brief explanation of why this specific music will help counteract the user's current emotional state",
  "song_description": "A detailed description of therapeutic music including genre, BPM, instruments, mood, and specific therapeutic intent"
}}

The song_description should be a single, detailed sentence suitable for Suno AI music generation that specifies the therapeutic musical elements needed to bring this user back to emotional baseline.
DO NOT ADD ANY LYRICS OR VOCALS TO THE SONG DESCRIPTION.
"""
        
        return prompt
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict of lists
            return obj.to_dict('list')
        elif isinstance(obj, pd.Series):
            # Convert Series to list
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj) or (isinstance(obj, float) and (obj != obj)):  # Check for NaN
            return None
        else:
            return obj
    
    
    def _parse_json_response(self, llm_response: str) -> Dict[str, str]:
        """Parse JSON response from LLM and extract song_reasoning and song_description."""
        try:
            # With response_format="json_object", the response should be direct JSON
            # But handle both cases: direct JSON or JSON in code blocks
            json_str = llm_response.strip()
            
            # Check if it's wrapped in code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Extract both fields
            result = {}
            if 'song_description' in parsed:
                result['song_description'] = parsed['song_description']
            else:
                result['song_description'] = "therapeutic ambient music at 75 BPM with calming instruments, designed to restore emotional balance"
            
            if 'song_reasoning' in parsed:
                result['song_reasoning'] = parsed['song_reasoning']
            else:
                result['song_reasoning'] = "The user needs calming music to restore emotional balance and reduce stress levels."
                
            return result
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {llm_response}")
            # Return fallback
            return {
                'song_reasoning': "The user needs calming music to restore emotional balance and reduce stress levels.",
                'song_description': "therapeutic ambient music at 75 BPM with calming instruments, designed to restore emotional balance"
            }
    
    def analyze_current_eeg_data(self, labeled_sample: Dict[str, Dict]) -> Dict[str, str]:
        """
        Analyze current EEG data sample and provide emotional analysis and current status.
        
        Args:
            labeled_sample: Dictionary containing labeled brainwave values with structure:
                           {"key": {"value": float, "label": str, "unit": str}}
            
        Returns:
            Dict[str, str]: Dictionary containing 'emotional_analysis' and 'current_status'
        """
        try:
            # Create the prompt for the LLM
            prompt = self._build_eeg_analysis_prompt(labeled_sample)
            
            # Make actual LLM API call
            llm_response = self.call_llm(prompt)
            
            # Parse JSON response
            analysis = self._parse_eeg_analysis_response(llm_response)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing current EEG data: {e}")
            # Fallback analysis
            fallback_analysis = self._fallback_eeg_analysis(labeled_sample)
            return {
                'emotional_analysis': fallback_analysis,
                'current_status': 'neutral'
            }
    
    def _build_eeg_analysis_prompt(self, labeled_sample: Dict[str, Dict]) -> str:
        """Build the prompt for analyzing current EEG data."""
        prompt = f"""You are a neuroscience AI that analyzes EEG brainwave data to determine current emotional state.

Current EEG Sample with Labels:
{json.dumps(labeled_sample, indent=2)}

Analyze these brainwave frequencies and provide a very short emotional analysis:
- Delta (0.5-4 Hz): Deep sleep, unconscious processes
- Theta (4-8 Hz): Creativity, meditation, REM sleep
- Low Alpha (8-10 Hz): Relaxed awareness, calm
- High Alpha (10-12 Hz): Alert relaxation, focused calm
- Low Beta (12-15 Hz): Focused attention, concentration
- High Beta (15-30 Hz): Alertness, anxiety, active thinking
- Low Gamma (30-40 Hz): Cognitive processing, consciousness
- Mid Gamma (40-100 Hz): High-level cognitive functions
- Attention: Focus level (0-100)
- Meditation: Relaxation level (0-100)

Respond with a JSON object containing only a short emotional analysis:

{{
  "emotional_analysis": "A brief 1-2 sentence analysis of the current emotional/mental state based on the brainwave patterns",
  "current_status": "Current emotion/mental status".
}}

Keep the analysis concise and focused on the dominant emotional state indicated by the brainwave patterns."""
        
        return prompt
    
    def _parse_eeg_analysis_response(self, llm_response: str) -> Dict[str, str]:
        """Parse JSON response from LLM and extract emotional analysis and current status."""
        try:
            json_str = llm_response.strip()
            
            # Check if it's wrapped in code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Extract both fields
            result = {}
            if 'emotional_analysis' in parsed:
                result['emotional_analysis'] = parsed['emotional_analysis']
            else:
                result['emotional_analysis'] = "Current brainwave patterns suggest a neutral emotional state with moderate alertness."
            
            if 'current_status' in parsed:
                result['current_status'] = parsed['current_status']
            else:
                result['current_status'] = "neutral"
                
            return result
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing EEG analysis response: {e}")
            print(f"Raw response: {llm_response}")
            # Return fallback
            return {
                'emotional_analysis': "Current brainwave patterns suggest a neutral emotional state with moderate alertness.",
                'current_status': "neutral"
            }
    
    def _fallback_eeg_analysis(self, labeled_sample: Dict[str, Dict]) -> str:
        """Provide fallback analysis based on EEG values."""
        attention = labeled_sample.get('attention', {}).get('value', 0)
        meditation = labeled_sample.get('meditation', {}).get('value', 0)
        high_beta = labeled_sample.get('highBeta', {}).get('value', 0)
        low_alpha = labeled_sample.get('lowAlpha', {}).get('value', 0)
        high_alpha = labeled_sample.get('highAlpha', {}).get('value', 0)
        theta = labeled_sample.get('theta', {}).get('value', 0)
        
        # Simple rule-based analysis
        if high_beta > 50000 and attention > 70:
            return "High alertness and focused attention detected, possibly indicating stress or intense concentration."
        elif meditation > 70 and (low_alpha + high_alpha) > 30000:
            return "Relaxed and meditative state with strong alpha waves, indicating calm awareness."
        elif theta > 40000 and attention < 40:
            return "Creative and introspective state with elevated theta waves, suggesting daydreaming or meditation."
        elif attention > 80 and meditation < 30:
            return "Highly focused but tense state, indicating active problem-solving or potential anxiety."
        elif meditation > 50 and attention > 50:
            return "Balanced state showing both focus and relaxation, indicating optimal mental performance."
        else:
            return "Neutral emotional state with moderate brain activity across all frequency bands."
    