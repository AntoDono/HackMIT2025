import os
import json
import re
from typing import Dict, Any
import pandas as pd
import numpy as np
from cerebras.cloud.sdk import Cerebras

class CerebrasClient:
    """Client for Cerebras API integration."""
    
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY')
        self.model = "gpt-oss-120b"  # Correct model name
        self.client = Cerebras(api_key=self.api_key) if self.api_key else None
    
    def call_llm(self, prompt: str) -> str:
        """Make actual API call to Cerebras LLM."""
        if not self.client:
            print("Warning: No CEREBRAS_API_KEY found, using mock response")
            return self._mock_llm_response_from_prompt(prompt)
        
        try:
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
                raise ValueError("No response content from Cerebras API")
                
        except Exception as e:
            print(f"Error calling Cerebras API: {e}")
            return self._mock_llm_response_from_prompt(prompt)
    
    def generate_song_description(self, classified_emotion: str, eeg_analysis: Dict[str, Any]) -> str:
        """
        Generate a therapeutic song description using LLM to bring user's emotion back to baseline.
        
        Args:
            classified_emotion: The classified emotional state of the user
            eeg_analysis: Dictionary containing EEG analysis results from eeg_stats.py
            
        Returns:
            str: A song description suitable for Suno AI music generation to restore emotional balance
        """
        try:
            # Create the prompt for the LLM
            prompt = self._build_therapeutic_prompt(classified_emotion, eeg_analysis)
            
            # Make actual LLM API call (falls back to mock if no API key)
            llm_response = self.call_llm(prompt)
            
            # Parse JSON response
            song_description = self._parse_json_response(llm_response)
            
            return song_description
            
        except Exception as e:
            print(f"Error generating song description: {e}")
            # Fallback description
            return "calm ambient music at 75 BPM with soft piano and strings, peaceful and relaxing atmosphere designed to restore emotional balance"
    
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
3. Therapeutic music principles for emotional regulation
4. Specific musical parameters (BPM, instruments, genre, mood) that promote emotional balance

Respond with a JSON object containing only the song_description field:

{{
  "song_description": "A detailed description of therapeutic music including genre, BPM, instruments, mood, and specific therapeutic intent"
}}

The song_description should be a single, detailed sentence suitable for Suno AI music generation that specifies the therapeutic musical elements needed to bring this user back to emotional baseline."""
        
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
    
    def _mock_llm_response(self, classified_emotion: str, eeg_analysis: Dict[str, Any]) -> str:
        """Mock LLM response for testing purposes."""
        # Extract key metrics for decision making
        cognitive = eeg_analysis.get('cognitive_indicators', {})
        focus_index = cognitive.get('focus_index', 0)
        relaxation_index = cognitive.get('relaxation_index', 0)
        mental_workload = cognitive.get('mental_workload', 0)
        
        # Generate therapeutic response based on emotion and EEG data
        if classified_emotion.lower() in ['stress', 'anxiety', 'nervous'] or mental_workload > 0.7:
            response = {
                "song_description": "Gentle ambient meditation track at 60-70 BPM with soft piano, warm strings, and subtle nature sounds, designed to reduce cortisol levels and activate the parasympathetic nervous system for deep relaxation and stress relief"
            }
        elif classified_emotion.lower() in ['sad', 'depressed', 'melancholy'] or (focus_index < 0.3 and relaxation_index < 0.3):
            response = {
                "song_description": "Uplifting acoustic folk track at 90-100 BPM with bright acoustic guitar, gentle percussion, and warm piano, crafted to gradually elevate mood and restore emotional balance through major key progressions and rhythmic stability"
            }
        elif classified_emotion.lower() in ['angry', 'frustrated', 'agitated'] or focus_index > 0.8:
            response = {
                "song_description": "Calming downtempo electronic track at 70-80 BPM with soft synthesizer pads, minimal beats, and flowing melodies, therapeutically designed to cool emotional intensity and guide toward centered tranquility"
            }
        elif classified_emotion.lower() in ['excited', 'manic', 'hyperactive']:
            response = {
                "song_description": "Soothing classical minimalism at 65-75 BPM featuring solo piano with sustained string accompaniment, structured to gently slow racing thoughts and restore emotional equilibrium through repetitive, calming patterns"
            }
        elif classified_emotion.lower() in ['focused', 'concentrated'] and relaxation_index < 0.4:
            response = {
                "song_description": "Balanced instrumental track at 80-85 BPM with acoustic guitar, soft strings, and light percussion, designed to maintain mental clarity while introducing gentle relaxation to prevent cognitive fatigue"
            }
        else:  # neutral or balanced state
            response = {
                "song_description": "Harmonious modern classical piece at 75-85 BPM with piano, strings, and subtle ambient textures, crafted to maintain emotional stability and support continued well-being"
            }
        
        return f"```json\n{json.dumps(response, indent=2)}\n```"
    
    def _mock_llm_response_from_prompt(self, prompt: str) -> str:
        """Generate mock response based on the prompt content."""
        # Extract emotion and EEG data from prompt for mock response
        if "stress" in prompt.lower() or "anxiety" in prompt.lower():
            response = {
                "song_description": "Gentle ambient meditation track at 60-70 BPM with soft piano, warm strings, and subtle nature sounds, designed to reduce cortisol levels and activate the parasympathetic nervous system for deep relaxation and stress relief"
            }
        elif "sad" in prompt.lower() or "depressed" in prompt.lower():
            response = {
                "song_description": "Uplifting acoustic folk track at 90-100 BPM with bright acoustic guitar, gentle percussion, and warm piano, crafted to gradually elevate mood and restore emotional balance through major key progressions and rhythmic stability"
            }
        elif "angry" in prompt.lower() or "frustrated" in prompt.lower():
            response = {
                "song_description": "Calming downtempo electronic track at 70-80 BPM with soft synthesizer pads, minimal beats, and flowing melodies, therapeutically designed to cool emotional intensity and guide toward centered tranquility"
            }
        elif "excited" in prompt.lower() or "manic" in prompt.lower():
            response = {
                "song_description": "Soothing classical minimalism at 65-75 BPM featuring solo piano with sustained string accompaniment, structured to gently slow racing thoughts and restore emotional equilibrium through repetitive, calming patterns"
            }
        else:
            response = {
                "song_description": "Harmonious modern classical piece at 75-85 BPM with piano, strings, and subtle ambient textures, crafted to maintain emotional stability and support continued well-being"
            }
        
        return f"```json\n{json.dumps(response, indent=2)}\n```"
    
    def _parse_json_response(self, llm_response: str) -> str:
        """Parse JSON response from LLM and extract song_description."""
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
            
            # Extract song_description
            if 'song_description' in parsed:
                return parsed['song_description']
            else:
                raise ValueError("No 'song_description' field found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {llm_response}")
            # Return fallback
            return "therapeutic ambient music at 75 BPM with calming instruments, designed to restore emotional balance"
    