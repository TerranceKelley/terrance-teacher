import json
import urllib.error
import urllib.request
from typing import Optional


class OllamaClient:
    """
    Client for Ollama local LLM API.
    
    Default model is llama3.2. Falls back to llama3 conceptually if needed
    (fallback logic not implemented - keep simple).
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text using Ollama API.
        
        Returns the response text on success, None on any error.
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                return response_data.get("response")
        
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError):
            return None

