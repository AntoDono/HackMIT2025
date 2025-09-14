import os
from typing import Dict, Any

class CerebrasClient:
    """Stub client for Cerebras API integration."""
    
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY')
        self.model = "Qwen3-235B-Thinking"  # User specified model
    
    def build_request(self, prompt: str) -> dict:
        """Build request payload for Cerebras API (no actual sending)."""
        return {
            "prompt": prompt,
            "model": self.model,
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
    
    def mock_response(self, prompt: str) -> str:
        """Return a canned response for testing purposes."""
        # Extract emotion and goal from prompt for context
        emotion = "unknown"
        goal = "calm"
        
        if "stress" in prompt.lower():
            emotion = "stress"
            goal = "calm"
        elif "relax" in prompt.lower():
            emotion = "relax"
            goal = "energize"
        elif "focused" in prompt.lower():
            emotion = "focused"
            goal = "maintain"
        elif "nervous" in prompt.lower():
            emotion = "nervous"
            goal = "calm"
        
        # Generate contextual mock response
        if goal == "calm":
            return """A gentle, flowing composition designed to soothe and center the mind. This piece features soft piano melodies with sustained string pads, creating a peaceful atmosphere that gradually slows the heart rate and promotes deep relaxation.

• Tempo: 65-75 BPM
• Key/Mode: C major, modal harmonies
• Timbre: Soft piano, warm strings, subtle pads
• Rhythm: Gentle, flowing patterns with minimal percussion
• Structure: Simple 8-bar phrases with gradual development
• Dynamics: Soft to mezzo-piano, with gentle swells"""
        
        elif goal == "energize":
            return """An uplifting, rhythmic piece that gently energizes without overwhelming. Bright acoustic guitar and light percussion create an optimistic mood that gradually increases energy and focus while maintaining a sense of calm.

• Tempo: 115-125 BPM
• Key/Mode: G major, bright and uplifting
• Timbre: Acoustic guitar, light drums, warm piano
• Rhythm: Clear, driving groove with syncopated accents
• Structure: 4-bar phrases with call-and-response patterns
• Dynamics: Moderate, building to a gentle peak"""
        
        elif goal == "maintain":
            return """A balanced composition that maintains current focus while gently softening any stress edges. Consistent piano and string textures provide stability with subtle variations to keep the mind engaged.

• Tempo: 85-95 BPM
• Key/Mode: F major, stable and centered
• Timbre: Piano, strings, minimal percussion
• Rhythm: Steady, predictable patterns
• Structure: 8-bar phrases with subtle variations
• Dynamics: Moderate, consistent levels"""
        
        else:
            return """A therapeutic musical piece designed to support emotional regulation and mental well-being. This composition uses carefully selected harmonic progressions and rhythmic patterns to create a supportive auditory environment.

• Tempo: 70-80 BPM
• Key/Mode: D major, warm and supportive
• Timbre: Piano, strings, gentle pads
• Rhythm: Flowing, organic patterns
• Structure: Simple, repetitive phrases
• Dynamics: Soft to moderate, with gentle variations"""