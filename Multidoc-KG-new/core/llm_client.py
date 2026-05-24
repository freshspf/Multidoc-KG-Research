"""
LLM client interface for LLM-as-a-Judge pattern using OpenAI API.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from urllib.parse import urlparse

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[LLMClient] Warning: openai library not installed. Standard OpenAI API calls will fail.")


# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """LLM client for validation and grounding tasks using OpenAI API."""
    
    def __init__(self, model_name: str = "gpt-4o", base_url: Optional[str] = None, timeout: int = 120):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the LLM model to use (default: gpt-4o)
            base_url: Optional base URL for OpenAI API (default: from env or OpenAI default)
            timeout: Request timeout in seconds (default: 120)
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Check if using custom API provider (non-standard endpoint)
        self.use_custom_api = self.base_url and "aigcbest" in self.base_url
        
        if self.use_custom_api:
            # Parse base URL for custom API
            parsed = urlparse(self.base_url)
            self.api_host = parsed.netloc
            self.api_scheme = parsed.scheme
            print(f"[LLMClient] Using custom API provider: {self.api_host}")
        else:
            # Initialize OpenAI client for standard API calls
            if OPENAI_AVAILABLE:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=timeout
                )
            else:
                self.client = None
                print("[LLMClient] Warning: OpenAI client not initialized. Install openai package for standard API calls.")
        
        print(f"[LLMClient] Initialized with model: {model_name}, base_url: {self.base_url or 'default'}, timeout: {timeout}s")
    
    def generate(self, prompt: str, system_prompt: str = "", json_mode: bool = False) -> str:
        """
        Generate text response from LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt for the conversation
            json_mode: Whether to force JSON output format
            
        Returns:
            Generated text response
        """
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Use custom API if configured
            if self.use_custom_api:
                return self._call_custom_api(messages, json_mode)
            
            # Standard OpenAI API call
            response_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "timeout": self.timeout,
            }
            
            if json_mode:
                response_kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**response_kwargs)
            
            result = response.choices[0].message.content
            print(f"[LLMClient] generate() completed, response length: {len(result) if result else 0}")
            return result or ""
            
        except Exception as e:
            print(f"[LLMClient] Error in generate(): {str(e)}")
            raise
    
    def _call_custom_api(self, messages: List[Dict], json_mode: bool = False) -> str:
        """
        Call custom API provider using requests library.
        
        Args:
            messages: List of message dictionaries
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text response
        """
        try:
            # Prepare URL
            url = f"{self.api_scheme}://{self.api_host}/v1/responses"
            
            # Prepare payload
            payload_dict = {
                "model": self.model_name,
                "input": messages,
            }
            
            if json_mode:
                payload_dict["response_format"] = {"type": "json_object"}
            
            # Prepare headers
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Make request with detailed logging
            print(f"[LLMClient] Calling custom API: {url}")
            print(f"[LLMClient] Model: {self.model_name}, Messages: {len(messages)}, Timeout: {self.timeout}s")
            
            response = requests.post(
                url,
                json=payload_dict,
                headers=headers,
                timeout=self.timeout,
                verify=True  # SSL verification
            )
            
            # Check response status
            print(f"[LLMClient] Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[LLMClient] Error response: {response.text[:500]}")
                response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            
            print(f"[LLMClient] DEBUG: Response type: {type(response_json)}")
            if isinstance(response_json, list) and len(response_json) > 0:
                print(f"[LLMClient] DEBUG: First item keys: {list(response_json[0].keys()) if isinstance(response_json[0], dict) else 'not a dict'}")
            elif isinstance(response_json, dict):
                print(f"[LLMClient] DEBUG: Response keys: {list(response_json.keys())}")
            
            # Extract content based on actual response format
            result = None
            
            # Format 1: Anthropic/Custom API format (list with nested content)
            # [{"id": "...", "type": "message", "content": [{"type": "output_text", "text": "..."}], "role": "assistant"}]
            if isinstance(response_json, list) and len(response_json) > 0:
                first_item = response_json[0]
                print(f"[LLMClient] DEBUG: Processing list format, first item type: {type(first_item)}")
                
                if isinstance(first_item, dict) and "content" in first_item:
                    content_list = first_item.get("content", [])
                    print(f"[LLMClient] DEBUG: Found content list with {len(content_list)} items")
                    
                    if isinstance(content_list, list) and len(content_list) > 0:
                        # Look for text field in content items
                        for idx, content_item in enumerate(content_list):
                            print(f"[LLMClient] DEBUG: Content item {idx}: {type(content_item)}, keys: {list(content_item.keys()) if isinstance(content_item, dict) else 'N/A'}")
                            if isinstance(content_item, dict) and "text" in content_item:
                                result = content_item["text"]
                                print(f"[LLMClient] SUCCESS: Extracted text from content[{idx}]['text']")
                                break
            
            # Format 2: Standard OpenAI format
            # {"choices": [{"message": {"content": "..."}}]}
            if result is None and isinstance(response_json, dict):
                print(f"[LLMClient] DEBUG: Trying standard OpenAI/dict format")
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    result = response_json["choices"][0].get("message", {}).get("content", "")
                    print(f"[LLMClient] SUCCESS: Extracted from choices[0].message.content")
                elif "output" in response_json:
                    output_value = response_json["output"]
                    print(f"[LLMClient] DEBUG: Found output field, type: {type(output_value)}")
                    
                    # Check if output is a list (nested structure)
                    if isinstance(output_value, list) and len(output_value) > 0:
                        first_output = output_value[0]
                        if isinstance(first_output, dict) and "content" in first_output:
                            content_list = first_output.get("content", [])
                            if isinstance(content_list, list) and len(content_list) > 0:
                                for content_item in content_list:
                                    if isinstance(content_item, dict) and "text" in content_item:
                                        result = content_item["text"]
                                        print(f"[LLMClient] SUCCESS: Extracted from output[0].content[0].text")
                                        break
                    
                    # If still None, treat output as direct value
                    if result is None:
                        result = output_value if isinstance(output_value, str) else str(output_value)
                        print(f"[LLMClient] SUCCESS: Using output field directly")
                elif "content" in response_json:
                    result = response_json["content"]
                    print(f"[LLMClient] SUCCESS: Extracted from content field")
                elif "text" in response_json:
                    result = response_json["text"]
                    print(f"[LLMClient] SUCCESS: Extracted from text field")
            
            # Fallback: convert entire response to string
            if result is None or result == "":
                print(f"[LLMClient] WARNING: Could not extract content, using fallback")
                print(f"[LLMClient] Full response structure: {json.dumps(response_json, indent=2)[:1000]}")
                result = json.dumps(response_json)
            
            # Ensure result is a string
            if isinstance(result, list):
                print(f"[LLMClient] Warning: Result is still a list, converting to string")
                result = json.dumps(result) if result else ""
            elif not isinstance(result, str):
                result = str(result)
            
            print(f"[LLMClient] Extracted content preview: {result[:150]}...")
            print(f"[LLMClient] Custom API call completed, response length: {len(result)}")
            return result
            
        except requests.exceptions.Timeout as e:
            print(f"[LLMClient] Request timeout after {self.timeout}s. URL: {url}")
            raise Exception(f"Request timeout: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            print(f"[LLMClient] Connection error. Cannot reach {self.api_host}")
            print(f"[LLMClient] Error details: {str(e)}")
            raise Exception(f"Connection error: Cannot reach API server. Please check network connection.")
        except requests.exceptions.RequestException as e:
            print(f"[LLMClient] Request error: {type(e).__name__}: {str(e)}")
            raise
        except Exception as e:
            print(f"[LLMClient] Unexpected error in custom API call: {type(e).__name__}: {str(e)}")
            raise
    
    def judge(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        LLM-as-a-Judge: Validate or evaluate a claim.
        
        Args:
            claim: The claim to judge
            context: Optional context information
            
        Returns:
            Dictionary with judgment results (e.g., {"valid": True, "confidence": 0.9})
        """
        try:
            system_prompt = "You are a knowledge validation judge. Evaluate claims for logical consistency and validity."
            user_prompt = f"Claim to evaluate: {claim}\n"
            if context:
                user_prompt += f"Context: {context}\n"
            user_prompt += "Respond with a JSON object containing 'valid' (boolean), 'confidence' (float 0-1), and 'reasoning' (string)."
            
            response = self.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            # Parse JSON response
            judgment = json.loads(response)
            print(f"[LLMClient] judge() completed for claim: {claim[:50]}...")
            return judgment
            
        except Exception as e:
            print(f"[LLMClient] Error in judge(): {str(e)}")
            return {"valid": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def ground_entities(self, entities: List[str], graph_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to help ground entities to existing graph nodes.
        
        Args:
            entities: List of entity strings to ground
            graph_context: Optional context about existing graph
            
        Returns:
            Dictionary mapping entities to potential graph node IDs
        """
        try:
            system_prompt = "You are an entity grounding assistant. Help match entities to existing graph nodes."
            user_prompt = f"Entities to ground: {', '.join(entities)}\n"
            if graph_context:
                user_prompt += f"Graph context: {graph_context}\n"
            user_prompt += "Respond with a JSON object mapping each entity to a potential node ID or null if no match."
            
            response = self.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            grounding_result = json.loads(response)
            print(f"[LLMClient] ground_entities() completed for {len(entities)} entities")
            return grounding_result
            
        except Exception as e:
            print(f"[LLMClient] Error in ground_entities(): {str(e)}")
            return {entity: None for entity in entities}


# Keep MockLLMClient for backward compatibility during transition
class MockLLMClient(LLMClient):
    """Mock LLM client for backward compatibility."""
    
    def __init__(self, model_name: str = "mock-llm"):
        """Initialize mock LLM client (uses real LLMClient but with mock model)."""
        print(f"[MockLLMClient] Using mock mode")
        # Don't call super().__init__() to avoid API initialization
        self.model_name = model_name
        print(f"[MockLLMClient] Initialized with model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method."""
        print(f"[MockLLMClient] generate() called with prompt length: {len(prompt)}")
        return "mock_response"
    
    def judge(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Mock judge method."""
        print(f"[MockLLMClient] judge() called for claim: {claim[:50]}...")
        return {"valid": True, "confidence": 0.9, "reasoning": "mock_reasoning"}
    
    def ground_entities(self, entities: List[str], graph_context: Optional[str] = None) -> Dict[str, Any]:
        """Mock ground_entities method."""
        print(f"[MockLLMClient] ground_entities() called for {len(entities)} entities")
        return {entity: f"mock_node_id_{i}" for i, entity in enumerate(entities)}
