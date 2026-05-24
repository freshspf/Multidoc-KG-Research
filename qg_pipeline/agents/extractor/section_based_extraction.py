#!/usr/bin/env python3
"""
Section-Based Knowledge Graph Extraction Test Script

This script uses the enhanced section-based extraction prompt to analyze papers
with special focus on formulas, model architectures, and examples.
"""

import os
import json
from openai import OpenAI
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Any

class SectionBasedExtractor:
    def __init__(self, output_dir: str = None, config: Dict = None):
        """Initialize the section-based extractor"""
        # Load configuration
        self.config = config or {}
        extractor_config = self.config.get('extractor', {})
        api_config = self.config.get('api', {})

        # Load API credentials from config first, then environment
        self.api_key = api_config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = api_config.get('openai_base_url') or os.getenv('OPENAI_BASE_URL')

        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量或在配置文件中指定")

        # Configure OpenAI client (new v1.0+ API)
        api_config = self.config.get('api', {})
        timeout = api_config.get('timeout', 600.0)  # 默认 10 分钟

        client_kwargs = {"api_key": self.api_key, "timeout": timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

        # Model configuration
        self.model = extractor_config.get('model', 'gpt-4o-mini')
        self.temperature = extractor_config.get('temperature', 0.1)
        self.max_tokens = extractor_config.get('max_tokens', 3000)

        # Set up paths
        script_dir = Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else script_dir.parent / "section_based_extractions"
        self.output_dir.mkdir(exist_ok=True)

        # Load the section-based prompt
        prompt_file = script_dir / 'section_based_extraction_prompt.txt'
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.section_prompt = f.read()
    
    def load_paper_chunks(self, paper_file: str) -> List[Dict]:
        """Load paper chunks from JSON file"""
        try:
            with open(paper_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"Cannot load paper file {paper_file}: {e}")
            return []
    
    def merge_paper_chunks(self, chunks: List[Dict]) -> str:
        """Merge paper chunks into full text with proper section labeling"""
        full_text = ""
        for i, chunk in enumerate(chunks):
            section_id = chunk.get('id', f'chunk_{i}')
            if 'text' in chunk:
                full_text += f"\n=== SECTION: {section_id} (CHUNK {i}) ===\n"
                full_text += chunk['text'] + "\n"
            elif 'content' in chunk:
                full_text += f"\n=== SECTION: {section_id} (CHUNK {i}) ===\n"
                full_text += chunk['content'] + "\n"
        return full_text.strip()
    
    def extract_paper_metadata(self, chunks: List[Dict]) -> Dict[str, str]:
        """Extract basic paper metadata"""
        metadata = {
            'title': 'Unknown Title',
            'authors': 'Unknown Authors',
            'chunks': len(chunks),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to extract title and authors from first chunk
        if chunks:
            first_chunk = chunks[0]
            text = first_chunk.get('text', first_chunk.get('content', ''))
            
            # Simple heuristics for title and authors
            lines = text.split('\n')[:10]  # Check first 10 lines
            for line in lines:
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    if any(word in line.lower() for word in ['learning', 'model', 'network', 'algorithm']):
                        metadata['title'] = line
                        break
        
        return metadata
    
    def call_openai_api(self, messages: List[Dict], max_tokens: int = None) -> str:
        """Call OpenAI API with error handling"""
        try:
            # Use provided max_tokens or default from config
            tokens = max_tokens if max_tokens is not None else self.max_tokens

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return f"API Error: {str(e)}"
    
    def extract_single_section(self, section_text: str, section_name: str, chunk_id: int, metadata: Dict, improvement_suggestions: str = "") -> str:
        """Extract knowledge from a single section"""
        
        # Prepare improvement guidance
        improvement_guidance = ""
        if improvement_suggestions:
            improvement_guidance = f"""

IMPROVEMENT GUIDANCE (from evaluator):
{improvement_suggestions}

Please incorporate these suggestions in your extraction to improve the quality of the knowledge graph."""
        
        system_prompt = f"""You are an expert knowledge extractor. Extract ALL entities, relationships, and facts from this academic paper section.

Section: {section_name} (Chunk {chunk_id})
Paper: {metadata['title']}

{self.section_prompt}

FOCUS FOR THIS SECTION:
- Extract EVERY entity mentioned (models, datasets, metrics, methods, authors, organizations)
- Extract EVERY relationship and comparison
- Extract ALL numerical values and scores
- Create comprehensive triples for multi-hop reasoning{improvement_guidance}

TARGET: 20-50 triples from this section alone."""

        user_prompt = f"""Extract MAXIMUM TRIPLES from this section of the academic paper.

SECTION: {section_name} (CHUNK {chunk_id})

SECTION CONTENT:
{section_text}

EXTRACTION REQUIREMENTS:
- Extract EVERY entity mentioned
- Extract EVERY relationship, comparison, evaluation
- Extract ALL numerical values with context
- Use :sourceChunk "{chunk_id}" and :sourceSection "{section_name}" for ALL triples{improvement_guidance}

OUTPUT FORMAT:
```turtle
# Entities from {section_name}
:EntityName rdf:type :EntityType ;
    :sourceChunk "{chunk_id}" ;
    :sourceSection "{section_name}" ;
    :contextText "original text snippet" .

# Relationships from {section_name}  
:Entity1 :relationshipType :Entity2 ;
    :sourceChunk "{chunk_id}" ;
    :sourceSection "{section_name}" ;
    :contextText "original text snippet" .
```

Be exhaustive - extract EVERY piece of factual information from this section."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"   Extracting section: {section_name} (Chunk {chunk_id})")
        return self.call_openai_api(messages, max_tokens=3000)

    def extract_section_knowledge(self, paper_text: str, metadata: Dict, improvement_suggestions: str = "") -> str:
        """Extract knowledge using section-by-section approach then merge"""
        
        print("Starting section-by-section extraction strategy...")
        
        if improvement_suggestions:
            print(f"Applying improvement suggestions: {improvement_suggestions}")
        
        # Parse sections from the merged text
        sections = self.parse_sections_from_text(paper_text)
        
        if not sections:
            print("Cannot parse sections, falling back to full paper extraction")
            return self.extract_full_paper(paper_text, metadata, improvement_suggestions)

        print(f"Found {len(sections)} sections")
        
        # Extract from each section individually
        all_knowledge_graphs = []
        
        for section_info in sections:
            section_name = section_info['name']
            chunk_id = section_info['chunk_id']
            content = section_info['content']
            
            print(f"   Processing: {section_name} (Length: {len(content)} characters)")
            
            # Extract knowledge from this section with improvement suggestions
            section_kg = self.extract_single_section(content, section_name, chunk_id, metadata, improvement_suggestions)
            
            if section_kg and "API Error" not in section_kg:
                all_knowledge_graphs.append(f"# ===== SECTION: {section_name.upper()} (CHUNK {chunk_id}) =====\n{section_kg}")
                print(f"   Extraction completed: {section_name}")
            else:
                print(f"   Extraction failed: {section_name}")
            
            # Brief pause between sections
            import time
            time.sleep(1)
        
        # Merge all knowledge graphs
        combined_kg = "\n\n".join(all_knowledge_graphs)
        
        print(f"Section-by-section extraction completed, processed {len(sections)} sections")
        return combined_kg
    
    def parse_sections_from_text(self, paper_text: str) -> List[Dict]:
        """Parse sections from the merged paper text"""
        sections = []
        lines = paper_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('=== SECTION:') and 'CHUNK' in line:
                # Save previous section
                if current_section:
                    sections.append({
                        'name': current_section['name'],
                        'chunk_id': current_section['chunk_id'],
                        'content': '\n'.join(current_content).strip()
                    })
                    current_content = []
                
                # Parse new section header
                # Format: === SECTION: SectionName (CHUNK X) ===
                parts = line.split('(CHUNK')
                if len(parts) == 2:
                    section_name = parts[0].replace('=== SECTION:', '').strip()
                    chunk_id = parts[1].split(')')[0].strip()
                    current_section = {'name': section_name, 'chunk_id': chunk_id}
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections.append({
                'name': current_section['name'],
                'chunk_id': current_section['chunk_id'],
                'content': '\n'.join(current_content).strip()
            })
        
        return sections
    
    def extract_full_paper(self, paper_text: str, metadata: Dict, improvement_suggestions: str = "") -> str:
        """Fallback: extract from full paper if section parsing fails"""
        print("Falling back to full paper extraction as a fallback...")
        
        # Prepare improvement guidance
        improvement_guidance = ""
        if improvement_suggestions:
            improvement_guidance = f"""

IMPROVEMENT GUIDANCE (from evaluator):
{improvement_suggestions}

Please incorporate these suggestions in your extraction to improve the quality of the knowledge graph."""
        
        system_prompt = f"""Extract knowledge from this academic paper for multi-hop reasoning.

Paper: {metadata['title']}
{self.section_prompt}{improvement_guidance}"""

        user_prompt = f"""Extract ALL entities and relationships from this paper:

{paper_text[:40000]}

Target: 100+ triples covering all sections.{improvement_guidance}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=4000)
    
    def save_extraction_results(self, paper_name: str, metadata: Dict, 
                              knowledge_graph: str, raw_text: str) -> Path:
        """Save extraction results in Turtle format only"""
        
        base_name = paper_name.replace('.json', '')
        
        # Save as Turtle/RDF format only
        turtle_file = self.output_dir / f"{base_name}_extraction.ttl"
        with open(turtle_file, 'w', encoding='utf-8') as f:
            f.write(knowledge_graph)
        
        print(f"Results saved: {turtle_file}")
        return turtle_file
    
    def save_extraction_results_with_evaluation(self, paper_name: str, metadata: Dict, 
                                              knowledge_graph: str, raw_text: str,
                                              evaluation_passed: bool, improvement_suggestions: str) -> Path:
        """Save extraction results with evaluation information"""
        
        base_name = paper_name.replace('.json', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create evaluation header
        evaluation_header = f"""# ===== EVALUATION RESULTS =====
# Evaluation Passed: {evaluation_passed}
# Can Proceed to QA Generation: {evaluation_passed}
# Timestamp: {timestamp}
"""
        
        if improvement_suggestions:
            evaluation_header += f"# Improvement Suggestions: {improvement_suggestions}\n"
        
        evaluation_header += "# ===== KNOWLEDGE GRAPH =====\n\n"
        
        # Save as Turtle/RDF format with evaluation info
        turtle_file = self.output_dir / f"{base_name}_extraction.ttl"
        with open(turtle_file, 'w', encoding='utf-8') as f:
            f.write(evaluation_header)
            f.write(knowledge_graph)
        
        # Also save evaluation metadata as JSON
        evaluation_file = self.output_dir / f"{base_name}_metadata.json"
        evaluation_data = {
            "paper_name": paper_name,
            "evaluation_passed": evaluation_passed,
            "can_proceed_to_qa": evaluation_passed,
            "improvement_suggestions": improvement_suggestions,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved: {turtle_file}")
        print(f"Evaluation information saved: {evaluation_file}")
        return turtle_file
    

    
    def process_paper(self, paper_file: str, evaluation_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single paper with section-based extraction"""
        
        print(f"Processing paper: {paper_file}")
        
        # Load paper data
        chunks = self.load_paper_chunks(paper_file)
        if not chunks:
            return {"error": "Cannot load paper file", "paper_file": paper_file}
        
        # Extract metadata
        metadata = self.extract_paper_metadata(chunks)
        
        # Merge chunks into full text
        full_text = self.merge_paper_chunks(chunks)
        metadata['text_length'] = len(full_text)
        
        print(f"Paper information:")
        print(f"   Title: {metadata['title']}")
        print(f"   Chunks: {metadata['chunks']}")
        print(f"   Text length: {metadata['text_length']} characters")
        
        # Get improvement suggestions from evaluation result
        improvement_suggestions = ""
        if evaluation_result and not evaluation_result.get("passed", True):
            improvement_suggestions = evaluation_result.get("suggestions", "")
        
        # Perform section-based extraction with improvement suggestions
        knowledge_graph = self.extract_section_knowledge(full_text, metadata, improvement_suggestions)
        
        # Process evaluation result
        evaluation_passed = True
        improvement_suggestions = ""
        
        if evaluation_result:
            evaluation_passed = evaluation_result.get("passed", False)
            improvement_suggestions = evaluation_result.get("suggestions", "")
            
            print(f"Evaluation results:")
            print(f"   Passed: {'Passed' if evaluation_passed else 'Failed'}")
            if improvement_suggestions:
                print(f"   Improvement suggestions: {improvement_suggestions}")
        
        # Save results with evaluation info
        paper_name = Path(paper_file).name
        saved_file = self.save_extraction_results_with_evaluation(
            paper_name, metadata, knowledge_graph, full_text, 
            evaluation_passed, improvement_suggestions
        )
        
        print(f"{paper_file} processing completed!")
        
        # Return results for workflow integration
        return {
            "success": True,
            "paper_file": paper_file,
            "metadata": metadata,
            "knowledge_graph": knowledge_graph,
            "saved_file": str(saved_file),
            "evaluation_passed": evaluation_passed,
            "improvement_suggestions": improvement_suggestions,
            "can_proceed_to_qa": evaluation_passed,  # 只有通过评估才能进行QA生成
            "statistics": {
                "total_triples": knowledge_graph.count(" ;") + knowledge_graph.count(" .") - knowledge_graph.count("# "),
                "models_extracted": knowledge_graph.count("rdf:type :Model"),
                "datasets_extracted": knowledge_graph.count("rdf:type :Dataset"),
                "metrics_extracted": knowledge_graph.count("rdf:type :Metric"),
                "methods_extracted": knowledge_graph.count("rdf:type :Method"),
                "sections_covered": len(set([line.split('"')[1] for line in knowledge_graph.split('\n') if ':sourceSection' in line])),
                "chunks_covered": len(set([line.split('"')[1] for line in knowledge_graph.split('\n') if ':sourceChunk' in line]))
            }
        }
    
    def process_single_paper(self, paper_file: str, evaluation_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single paper file - main interface for workflow integration"""
        if not Path(paper_file).exists():
            return {"error": f"File not found: {paper_file}", "paper_file": paper_file}
        
        return self.process_paper(paper_file, evaluation_result)
    
    def process_paper_from_text(self, paper_text: str, paper_name: str = "unknown", 
                               evaluation_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process paper from text content directly"""
        print(f"Processing text content: {paper_name}")
        
        # Create metadata
        metadata = {
            'title': paper_name,
            'authors': 'Unknown',
            'chunks': 1,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(paper_text)
        }
        
        # Get improvement suggestions from evaluation result
        improvement_suggestions = ""
        if evaluation_result and not evaluation_result.get("passed", True):
            improvement_suggestions = evaluation_result.get("suggestions", "")
        
        # Perform extraction with improvement suggestions
        knowledge_graph = self.extract_section_knowledge(paper_text, metadata, improvement_suggestions)
        
        # Process evaluation result
        evaluation_passed = True
        improvement_suggestions = ""
        
        if evaluation_result:
            evaluation_passed = evaluation_result.get("passed", True)
            improvement_suggestions = evaluation_result.get("suggestions", "")
        
        # Save results with evaluation info
        saved_file = self.save_extraction_results_with_evaluation(
            f"{paper_name}.txt", metadata, knowledge_graph, paper_text,
            evaluation_passed, improvement_suggestions
        )
        
        return {
            "success": True,
            "paper_name": paper_name,
            "metadata": metadata,
            "knowledge_graph": knowledge_graph,
            "saved_file": str(saved_file),
            "evaluation_passed": evaluation_passed,
            "improvement_suggestions": improvement_suggestions,
            "can_proceed_to_qa": evaluation_passed,
            "statistics": {
                "total_triples": knowledge_graph.count(" ;") + knowledge_graph.count(" .") - knowledge_graph.count("# "),
                "models_extracted": knowledge_graph.count("rdf:type :Model"),
                "datasets_extracted": knowledge_graph.count("rdf:type :Dataset"),
                "metrics_extracted": knowledge_graph.count("rdf:type :Metric"),
                "methods_extracted": knowledge_graph.count("rdf:type :Method")
            }
        }

def main():
    """Main execution function - for testing the extractor with evaluation"""
    print("Section-based knowledge graph extraction test program")
    print("=" * 50)
    
    # Initialize extractor
    try:
        extractor = SectionBasedExtractor()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Example 1: Process a single paper file with evaluation result
    paper_file = "/home/shenxiaoli/getKG-schema/kg_pipeline/data/data_test/papers_0_merged.json"
    
    # Simulate evaluation result from evaluator
    evaluation_result = {
        "passed": True,  # 评估通过
        "suggestions": "Please increase the number of entity relationships to improve the completeness of the graph"
    }
    
    result = extractor.process_single_paper(paper_file, evaluation_result)
    
    if result.get("success"):
        print(f"Test successful!")
        print(f"   Triples: {result['statistics']['total_triples']}")
        print(f"   Models: {result['statistics']['models_extracted']}")
        print(f"   Datasets: {result['statistics']['datasets_extracted']}")
        print(f"   Evaluation passed: {result['evaluation_passed']}")
        print(f"   Can proceed to QA generation: {result['can_proceed_to_qa']}")
        if result['improvement_suggestions']:
            print(f"   Improvement suggestions: {result['improvement_suggestions']}")
        print(f"   Saved file: {result['saved_file']}")
    else:
        print(f"Test failed: {result.get('error')}")
    
    print(f"\nProgram executed successfully! Results saved in: {extractor.output_dir}")

def test_with_failed_evaluation():
    """Test with failed evaluation result"""
    print("\nTest with failed evaluation result")
    print("-" * 30)
    
    extractor = SectionBasedExtractor()
    
    # Simulate failed evaluation with specific improvement suggestions
    failed_evaluation = {
        "passed": False,
        "suggestions": "Please increase the number of entity relationships to improve the completeness of the graph, especially focusing on the comparison relationships between models and performance indicators"
    }
    
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    Convolutional Neural Networks (CNNs) are commonly used for image processing tasks.
    Recurrent Neural Networks (RNNs) are designed to work with sequential data.
    ResNet achieved 3.57% top-5 error on ImageNet, outperforming previous models.
    """
    
    result = extractor.process_paper_from_text(sample_text, "test_paper", failed_evaluation)
    
    if result.get("success"):
        print(f"Processing completed!")
        print(f"   Evaluation passed: {result['evaluation_passed']}")
        print(f"   Can proceed to QA generation: {result['can_proceed_to_qa']}")
        print(f"   Improvement suggestions: {result['improvement_suggestions']}")
        print(f"   Triples: {result['statistics']['total_triples']}")
    else:
        print(f"Processing failed: {result.get('error')}")

def test_improvement_suggestions_impact():
    """Test how improvement suggestions affect extraction quality"""
    print("\nTest how improvement suggestions affect extraction quality")
    print("-" * 40)
    
    extractor = SectionBasedExtractor()
    
    sample_text = """
    Machine learning is a subset of artificial intelligence.
    Deep learning uses neural networks for complex tasks.
    CNNs are effective for image processing.
    RNNs handle sequential data well.
    """
    
    # Test 1: Without improvement suggestions
    print("Test 1: Without improvement suggestions")
    result1 = extractor.process_paper_from_text(sample_text, "test1_no_suggestions")
    triples1 = result1['statistics']['total_triples']
    print(f"   Triples: {triples1}")
    
    # Test 2: With improvement suggestions
    print("\nTest 2: With improvement suggestions")
    improvement_evaluation = {
        "passed": False,
        "suggestions": "Please focus on the comparison relationships between models, extract more performance indicators and evaluation results, and increase the connection relationships between entities"
    }
    
    result2 = extractor.process_paper_from_text(sample_text, "test2_with_suggestions", improvement_evaluation)
    triples2 = result2['statistics']['total_triples']
    print(f"   Triples: {triples2}")
    print(f"   Improvement suggestions: {result2['improvement_suggestions']}")
    
    # Compare results
    print(f"\nComparison results:")
    print(f"   Without suggestions: {triples1} triples")
    print(f"   With suggestions: {triples2} triples")
    print(f"   Improvement effect: {triples2 - triples1} triples")

def example_workflow_usage():
    """Example of how to use this extractor in a workflow"""
    print("\nWorkflow integration example:")
    print("-" * 30)
    
    # Initialize extractor
    extractor = SectionBasedExtractor(output_dir="workflow_results")
    
    # Example 1: Process a single paper file (file path would come from workflow)
    paper_file = "/path/to/your/paper.json"  # This would be passed from workflow
    result = extractor.process_single_paper(paper_file)
    
    # Example 2: Process text content directly (text would come from workflow)
    paper_text = """
    This is a sample paper about machine learning.
    The paper discusses various algorithms and their performance.
    """
    result2 = extractor.process_paper_from_text(paper_text, "sample_paper")
    
    print("Workflow integration examples completed!")

if __name__ == "__main__":
    main()
    test_with_failed_evaluation()
    test_improvement_suggestions_impact()