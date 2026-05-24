"""
Extraction Agent: Chunks papers and extracts Knowledge Claims.
"""
import json
import re
from typing import List
from schema import Paper, KnowledgeClaim, ClaimStatus
from core.llm_client import LLMClient


# Updated Schema for Medical/Biomedical Context
ONTOLOGY_SCHEMA = """
**ALLOWED CLASSES (PascalCase):**
- **Anatomy**: Body parts, brain regions (e.g., `PrefrontalCortex`, `VisualCortex`, `Hand`).
- **PhysiologicalProcess**: Biological functions (e.g., `AlphaRhythm`, `HeartRate`, `NeuralOscillation`).
- **Symptom**: Clinical signs or patient complaints (e.g., `Fatigue`, `Headache`).
- **Condition**: Diseases, disorders, or syndromes (e.g., `Epilepsy`, `Stroke`).
- **Treatment**: Drugs, surgeries, therapies (e.g., `Aspirin`, `DeepBrainStimulation`).
- **Method**: Diagnostic or experimental techniques (e.g., `EEG`, `MRI`, `SSVEP`).
- **Metric**: Quantifiable measurements (e.g., `Accuracy`, `Amplitude`, `P-Value`).
- **Tool**: Medical devices or software (e.g., `EmotivEpoc`, `Catheter`).
- **Finding**: Observed correlations or results (e.g., `IncreasedActivity`, `Correlation`).

**ALLOWED RELATIONS (camelCase):**
- rdf:type (Map entity to Class, e.g., "FrontalLobe" rdf:type "Anatomy")
- locatedIn (Anatomy -> Anatomy)
- affects (Condition/Treatment -> Anatomy/PhysiologicalProcess)
- manifestsAs (Condition -> Symptom)
- treats (Treatment -> Condition)
- indicates (Symptom/Finding -> Condition)
- measuredBy (PhysiologicalProcess -> Method)
- hasValue (Metric -> Value)
- associatedWith (Generic correlation)
"""


class ExtractionAgent:
    """Agent responsible for extracting knowledge claims from papers."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize Extraction Agent.
        
        Args:
            llm_client: LLM client for claim extraction
        """
        self.llm_client = llm_client
        print("[ExtractionAgent] Initialized")
    
    def _chunk_paper(self, paper: Paper, max_chunk_size: int = 8000) -> List[str]:
        """
        Chunk paper content into smaller pieces for processing.
        
        For now, implements simple splitting logic:
        - If content is short enough, treat as one chunk
        - Otherwise, split by paragraphs
        
        Args:
            paper: Paper object to chunk
            max_chunk_size: Maximum size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        content = paper.content
        
        # If content is short enough, return as single chunk
        if len(content) <= max_chunk_size:
            print(f"[ExtractionAgent] Paper content fits in one chunk ({len(content)} chars)")
            return [content]
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed max size, save current chunk and start new one
            if current_chunk and len(current_chunk) + len(para) + 2 > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"[ExtractionAgent] Split paper into {len(chunks)} chunks")
        return chunks
    
    def _extract_claims_from_chunk(self, chunk: str, paper_id: str, chunk_idx: int) -> List[KnowledgeClaim]:
        """
        Extract knowledge claims from a single chunk using LLM.
        
        Args:
            chunk: Text chunk to process
            paper_id: ID of the source paper
            chunk_idx: Index of the chunk (for logging)
            
        Returns:
            List of extracted KnowledgeClaim objects
        """
        system_prompt = (
            "You are a strict Scientific Knowledge Graph Extractor. "
            "Your goal is to extract atomic, structured triples (subject, relation, object) that align with a standard Ontology.\n\n"
            
            "**CRITICAL ENTITY RULES (Prevent Long Entities):**\n"
            "1. **Atomic Noun Phrases**: Entities MUST be single nouns or precise noun phrases (e.g., 'EEG Signal', 'Accuracy').\n"
            "2. **No Verbs/Clauses**: NEVER include verbs, relative clauses ('which is...'), or prepositions describing a process in the entity name.\n"
            "   - BAD: 'subtracting the baseline mean from every time point'\n"
            "   - GOOD: 'BaselineMeanSubtraction'\n"
            "   - BAD: 'visual-evoked amplitude increases with proximity'\n"
            "   - GOOD: (VisualEvokedAmplitude, increasesWith, Proximity)\n"
            "3. **Split Compound Entities**: If text says 'high portability and powerful computation', extract TWO triples.\n"
            "4. **Isolate Values**: Do not bake values into the metric name.\n"
            "   - BAD: 'Accuracy of 83%'\n"
            "   - GOOD: (Accuracy, hasValue, '83%')\n\n"
            
            "**ONTOLOGY NAMING CONVENTIONS:**\n"
            "- **Classes**: PascalCase (e.g., `Method`, `Metric`, `Tool`).\n"
            "- **Relations**: camelCase (e.g., `measuredBy`, `hasAttribute`, `increasedBy`).\n"
            "- **Relation Specificity**: Avoid generic 'has', 'is'. Use specific scientific verbs.\n\n"
            "**ONTOLOGY_SCHEMA (Medical/Biomedical):**\n"
            f"{ONTOLOGY_SCHEMA}"
        )
        
        user_prompt = (
            f"Extract atomic knowledge triples from the text below.\n\n"
            
            f"**STRICT FORMATTING EXAMPLES:**\n"
            f"Text: 'The proposed SSVEP-BCI system achieved an accuracy of 95%.'\n"
            f"Output: [\n"
            f"  {{'subject': 'SSVEP-BCI System', 'relation': 'rdf:type', 'object': 'System'}},\n"
            f"  {{'subject': 'SSVEP-BCI System', 'relation': 'hasMetric', 'object': 'Accuracy'}},\n"
            f"  {{'subject': 'Accuracy', 'relation': 'hasValue', 'object': '95%'}}\n"
            f"]\n\n"
            
            f"**NEGATIVE CONSTRAINTS (DO NOT DO THIS):**\n"
            f"- DO NOT extract sentences as entities (e.g., 'system performs well in noise').\n"
            f"- DO NOT include citations (e.g., '[9]') in entity names.\n"
            f"- DO NOT use 'is a type of' or 'includes' as relations; use `rdf:type` or `hasPart`.\n\n"
            f"- DO NOT extract patient states as long sentences (e.g., 'Patient feels pain in leg'). Extract as (Patient, hasSymptom, LegPain) or (LegPain, locatedIn, Leg).\n"
            f"- DO NOT confuse 'Finding' with 'Fact'. A Finding must be a specific scientific observation (e.g., 'GammaBandIncrease'), not a general statement.\n\n"
            
            f"Text to extract from:\n{chunk}\n\n"
            
            f"Return a JSON object with a 'claims' field:\n"
            f"{{\n"
            f"  \"claims\": [\n"
            f"    {{\n"
            f"      \"subject\": \"AtomicEntityName\",\n"
            f"      \"relation\": \"camelCaseRelation\",\n"
            f"      \"object\": \"AtomicEntityName OR Value\",\n"
            f"      \"evidence\": \"exact short quote\"\n"
            f"    }}\n"
            f"  ]\n"
            f"}}"
        )
        
        try:
            print(f"[ExtractionAgent] Extracting claims from chunk {chunk_idx + 1} ({len(chunk)} chars)...")
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            # Clean response - remove markdown code blocks if present
            # Ensure response is a string (defensive check)
            if isinstance(response, list):
                print(f"[ExtractionAgent] Warning: Response is a list, converting to JSON string")
                response = json.dumps(response)
            elif not isinstance(response, str):
                response = str(response)
            
            # More aggressive markdown cleaning
            response_clean = response.strip()
            
            # Remove markdown code blocks (```json ... ``` or ``` ... ```)
            # Use regex to handle newlines in markdown delimiters
            # Pattern: ```json\n ... \n```  or  ```\n ... \n```
            code_block_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```\s*$'
            match = re.match(code_block_pattern, response_clean, re.DOTALL)
            if match:
                response_clean = match.group(1).strip()
            else:
                # Fallback: simple string-based removal
                if response_clean.startswith("```json"):
                    response_clean = response_clean[7:].strip()
                elif response_clean.startswith("```"):
                    response_clean = response_clean[3:].strip()
                if response_clean.endswith("```"):
                    response_clean = response_clean[:-3].strip()
            
            print(f"[ExtractionAgent] Cleaned response preview: {response_clean[:150]}...")
            
            # Parse JSON response
            try:
                # Try parsing as JSON object first (some models wrap arrays in objects)
                parsed = json.loads(response_clean)
                if isinstance(parsed, dict):
                    # Look for common keys that might contain the array
                    for key in ["claims", "results", "data", "items"]:
                        if key in parsed and isinstance(parsed[key], list):
                            parsed = parsed[key]
                            break
                    # If still a dict, try to find any list value
                    if isinstance(parsed, dict):
                        for value in parsed.values():
                            if isinstance(value, list):
                                parsed = value
                                break
                
                # Ensure we have a list
                if not isinstance(parsed, list):
                    print(f"[ExtractionAgent] Warning: Expected list, got {type(parsed)}, returning empty list")
                    return []
                
                # Convert to KnowledgeClaim objects
                claims = []
                for item in parsed:
                    try:
                        claim = KnowledgeClaim(
                            subject=item.get("subject", ""),
                            relation=item.get("relation", ""),
                            object=item.get("object", ""),
                            evidence=item.get("evidence", ""),
                            source_paper_id=paper_id,
                            status=ClaimStatus.EXTRACTED
                        )
                        claims.append(claim)
                    except Exception as e:
                        print(f"[ExtractionAgent] Warning: Failed to parse claim item: {str(e)}")
                        continue
                
                print(f"[ExtractionAgent] Successfully extracted {len(claims)} claims from chunk {chunk_idx + 1}")
                return claims
                
            except json.JSONDecodeError as e:
                print(f"[ExtractionAgent] Error: Failed to parse JSON from chunk {chunk_idx + 1}: {str(e)}")
                print(f"[ExtractionAgent] Response preview: {response_clean[:200]}...")
                return []
                
        except Exception as e:
            print(f"[ExtractionAgent] Error extracting claims from chunk {chunk_idx + 1}: {str(e)}")
            return []
    
    def process(self, paper: Paper) -> List[KnowledgeClaim]:
        """
        Process a paper and extract knowledge claims.
        
        Args:
            paper: Paper object to process
            
        Returns:
            List of extracted KnowledgeClaim objects
        """
        print(f"[ExtractionAgent] Processing paper: {paper.id} - {paper.title}")
        
        # Step 1: Chunk the paper
        chunks = self._chunk_paper(paper)
        
        # Step 2: Extract claims from each chunk
        all_claims = []
        for idx, chunk in enumerate(chunks):
            chunk_claims = self._extract_claims_from_chunk(chunk, paper.id, idx)
            all_claims.extend(chunk_claims)
        
        print(f"[ExtractionAgent] Total extracted {len(all_claims)} claims from paper {paper.id}")
        return all_claims
