#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import random
from typing import List, Dict, Tuple, Set
from pathlib import Path
from openai import OpenAI
from datetime import datetime

class MultiHopQAGenerator:
    def __init__(self, openai_api_key: str = None, base_url: str = None, config: Dict = None):
        """
        Initialize the multi-hop QA generator

        Args:
            openai_api_key: OpenAI API key (optional if in config)
            base_url: Optional OpenAI base URL (optional if in config)
            config: Configuration dictionary
        """
        # Load configuration
        self.config = config or {}
        qa_config = self.config.get('qa_generator', {})
        api_config = self.config.get('api', {})

        # Get API key and base URL from config first, then parameters, then environment
        api_key = api_config.get('openai_api_key') or openai_api_key or os.getenv('OPENAI_API_KEY')
        base_url = api_config.get('openai_base_url') or base_url or os.getenv('OPENAI_BASE_URL')

        if not api_key:
            raise ValueError("OpenAI API key is required. Set in config, environment variable, or pass parameter.")

        # Initialize OpenAI client
        api_config = self.config.get('api', {})
        timeout = api_config.get('timeout', 600.0)  # 默认 10 分钟

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

        # Model configuration
        self.model = qa_config.get('model', 'gpt-4o-mini')
        self.temperature = qa_config.get('temperature', 0.7)
        self.max_tokens = qa_config.get('max_tokens', 800)

        self.hop_paths = []
        self.qa_pairs = []
        
    def parse_ttl_file(self, ttl_file_path: str) -> Dict[str, List[Dict]]:
        """
        Parse TTL file and extract entities and relationships by section
        
        Args:
            ttl_file_path: Path to the TTL file
            
        Returns:
            Dictionary mapping section names to their triples
        """
        sections = {}
        
        with open(ttl_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by section markers - handle different formats
        section_splits = re.split(r'# ===== SECTION: (.+?) (?:\(CHUNK \d+\) )?=====', content)
        
        # Process sections
        for i in range(1, len(section_splits), 2):
            if i + 1 < len(section_splits):
                section_name = section_splits[i].strip()
                section_content = section_splits[i + 1]
                
                # Clean section name
                section_name = re.sub(r'_\d+$', '', section_name)  # Remove _1, _2 suffixes
                
                # Extract turtle content
                turtle_pattern = r'```turtle\n(.*?)```'
                turtle_matches = re.findall(turtle_pattern, section_content, re.DOTALL)
                
                if turtle_matches:
                    for turtle_content in turtle_matches:
                        triples = self.extract_triples_from_section(turtle_content, section_name)
                        if section_name not in sections:
                            sections[section_name] = []
                        sections[section_name].extend(triples)
                else:
                    # Try to extract triples from raw content
                    triples = self.extract_triples_from_section(section_content, section_name)
                    if triples:
                        if section_name not in sections:
                            sections[section_name] = []
                        sections[section_name].extend(triples)
            
        print(f"Parsing completed, found {len(sections)} sections")
        for section, triples in sections.items():
            print(f"  {section}: {len(triples)} triples")
        return sections
    
    def extract_triples_from_section(self, section_content: str, section_name: str) -> List[Dict]:
        """
        Extract triples from a section's turtle content
        
        Args:
            section_content: The turtle content of the section
            section_name: Name of the section
            
        Returns:
            List of triple dictionaries
        """
        triples = []
        lines = section_content.split('\n')
        current_triple = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Match main triple patterns
            # Pattern 1: :Subject :predicate :Object ;
            main_triple_match = re.search(r':(\w+(?:-\w+)*)\s+:(\w+)\s+:(\w+(?:-\w+)*)\s*;', line)
            if main_triple_match:
                subject = main_triple_match.group(1)
                predicate = main_triple_match.group(2)
                object_val = main_triple_match.group(3)
                
                current_triple = {
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_val,
                    'section': section_name,
                    'chunk': '',
                    'context': ''
                }
                continue
            
            # Pattern 2: :Subject rdf:type :Type ;
            type_match = re.search(r':(\w+(?:-\w+)*)\s+rdf:type\s+:(\w+(?:-\w+)*)\s*;', line)
            if type_match:
                subject = type_match.group(1)
                object_val = type_match.group(2)
                
                current_triple = {
                    'subject': subject,
                    'predicate': 'type',
                    'object': object_val,
                    'section': section_name,
                    'chunk': '',
                    'context': ''
                }
                continue
            
            # Extract metadata
            if current_triple:
                # Source chunk
                chunk_match = re.search(r':sourceChunk\s+"(\d+)"', line)
                if chunk_match:
                    current_triple['chunk'] = chunk_match.group(1)
                
                # Source section
                section_match = re.search(r':sourceSection\s+"([^"]+)"', line)
                if section_match:
                    current_triple['section'] = section_match.group(1)
                
                # Context text
                context_match = re.search(r':contextText\s+"([^"]+)"', line)
                if context_match:
                    current_triple['context'] = context_match.group(1)
                    # Triple is complete, add it
                    if current_triple['subject'] and current_triple['predicate'] and current_triple['object']:
                        triples.append(current_triple.copy())
                    current_triple = {}
        
        return triples
    
    def find_3hop_paths(self, triples: List[Dict], max_paths_per_section: int = 10) -> List[Dict]:
        """
        Find 3-hop paths in the knowledge graph
        
        Args:
            triples: List of all triples
            max_paths_per_section: Maximum paths to find per section
            
        Returns:
            List of 3-hop path dictionaries
        """
        # Group triples by section
        section_triples = {}
        for triple in triples:
            section = triple['section']
            if section not in section_triples:
                section_triples[section] = []
            section_triples[section].append(triple)
        
        # Build adjacency graph for each section
        all_paths = []
        
        for section, section_triple_list in section_triples.items():
            print(f"Analyzing section: {section}")
            
            # Build graph
            graph = {}
            for triple in section_triple_list:
                subj = triple['subject']
                obj = triple['object']
                pred = triple['predicate']
                
                if subj not in graph:
                    graph[subj] = []
                graph[subj].append({
                    'target': obj,
                    'relation': pred,
                    'context': triple['context'],
                    'chunk': triple['chunk']
                })
            
            # Find 3-hop paths
            section_paths = self.find_paths_in_graph(graph, section, max_paths_per_section)
            all_paths.extend(section_paths)
            
            print(f"   Found {len(section_paths)} 3-hop paths")
        
        print(f"Total 3-hop paths found: {len(all_paths)}")
        return all_paths
    
    def find_paths_in_graph(self, graph: Dict, section: str, max_paths: int) -> List[Dict]:
        """
        Find 3-hop paths in a single graph
        
        Args:
            graph: Adjacency graph representation
            section: Section name
            max_paths: Maximum paths to find
            
        Returns:
            List of path dictionaries
        """
        paths = []
        nodes = list(graph.keys())
        
        for start_node in nodes:
            if len(paths) >= max_paths:
                break
                
            # Get 1-hop neighbors
            if start_node not in graph:
                continue
                
            for hop1 in graph[start_node]:
                node1 = hop1['target']
                if node1 not in graph:
                    continue
                    
                # Get 2-hop neighbors
                for hop2 in graph[node1]:
                    node2 = hop2['target']
                    if node2 not in graph:
                        continue
                        
                    # Get 3-hop neighbors
                    for hop3 in graph[node2]:
                        node3 = hop3['target']
                        
                        # Avoid cycles and self-loops
                        if node3 != start_node and node3 != node1 and node3 != node2:
                            path = {
                                'section': section,
                                'start_entity': start_node,
                                'end_entity': node3,
                                'hops': [
                                    {
                                        'from': start_node,
                                        'relation': hop1['relation'],
                                        'to': node1,
                                        'context': hop1['context'],
                                        'chunk': hop1['chunk']
                                    },
                                    {
                                        'from': node1,
                                        'relation': hop2['relation'],
                                        'to': node2,
                                        'context': hop2['context'],
                                        'chunk': hop2['chunk']
                                    },
                                    {
                                        'from': node2,
                                        'relation': hop3['relation'],
                                        'to': node3,
                                        'context': hop3['context'],
                                        'chunk': hop3['chunk']
                                    }
                                ]
                            }
                            paths.append(path)
                            
                            if len(paths) >= max_paths:
                                return paths
        
        return paths
    
    def generate_qa_pairs(self, paths: List[Dict], max_qa_per_section: int = 5) -> List[Dict]:
        """
        Generate multiple-choice QA pairs based on 3-hop paths
        
        Args:
            paths: List of 3-hop paths
            max_qa_per_section: Maximum QA pairs per section
            
        Returns:
            List of QA pair dictionaries
        """
        # Group paths by section
        section_paths = {}
        for path in paths:
            section = path['section']
            if section not in section_paths:
                section_paths[section] = []
            section_paths[section].append(path)
        
        all_qa_pairs = []
        
        for section, section_path_list in section_paths.items():
            print(f"Generating QA pairs for section: {section}")
            
            # Select representative paths
            selected_paths = random.sample(
                section_path_list, 
                min(max_qa_per_section, len(section_path_list))
            )
            
            for i, path in enumerate(selected_paths):
                print(f"   Generating {i+1}/{len(selected_paths)} QA pairs")
                
                qa_pair = self.generate_single_qa(path, section)
                if qa_pair:
                    all_qa_pairs.append(qa_pair)
        
        print(f"Total QA pairs generated: {len(all_qa_pairs)}")
        return all_qa_pairs
    
    def generate_single_qa(self, path: Dict, section: str) -> Dict:
        """
        Generate a single multiple-choice QA pair using OpenAI API
        
        Args:
            path: 3-hop path dictionary
            section: Section name
            
        Returns:
            QA pair dictionary
        """
        # Construct path description
        path_description = f"Path: {path['start_entity']}"
        contexts = []
        
        for hop in path['hops']:
            path_description += f" --[{hop['relation']}]--> {hop['to']}"
            contexts.append(hop['context'])
        
        context_text = " | ".join(contexts)
        
        system_prompt = """You are an expert at creating educational multiple-choice questions based on academic paper knowledge graphs. 

Your task is to generate ONE high-quality multiple-choice question with 4 options (A, B, C, D) based on a 3-hop reasoning path from a knowledge graph.

Requirements:
1. The question should require multi-hop reasoning following the provided path
2. Create exactly 4 answer options with only ONE correct answer
3. Options should be plausible but clearly distinguishable
4. Question should be academic and precise
5. Base the question on the provided context from the paper section
6. Make the question challenging but answerable from the given information
7. Write in English, and keep the questions objective
8. Whenever possible, pick phenomena in the paper that differ from common knowledge and are more surprising (this requires consulting the paper)
9. You may use a term that a domain expert would understand but that is likely ambiguous for an LLM; this ambiguous term should not appear elsewhere in the text
10. Each option must be longer than 10 words
11. Internal–external integration: the question must incorporate internal knowledge that does not appear in the paper
12. Wrong options should be somewhat misleading (e.g., parts of their wording overlap with the question)

Output format (JSON):
{
    "question": "Clear, specific question text",
    "options": {
        "A": "First option (must be longer than 10 words)",
        "B": "Second option (must be longer than 10 words)", 
        "C": "Third option (must be longer than 10 words)",
        "D": "Fourth option (must be longer than 10 words)"
    },
    "correct_answer": "A/B/C/D",
    "explanation": "Brief explanation of why the answer is correct"
}"""

        user_prompt = f"""Based on the following knowledge graph path and context from the paper section "{section}", generate ONE multiple-choice question:

KNOWLEDGE PATH:
{path_description}

CONTEXT FROM PAPER:
{context_text}

SECTION: {section}

Generate a multiple-choice question that requires reasoning through this 3-hop path. The question should test understanding of the relationships and entities involved.

Respond with valid JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                qa_data = json.loads(result)
                qa_pair = {
                    'section': section,
                    'path': path,
                    'question': qa_data['question'],
                    'options': qa_data['options'],
                    'correct_answer': qa_data['correct_answer'],
                    'explanation': qa_data['explanation'],
                    'path_description': path_description,
                    'context': context_text
                }
                return qa_pair
                
            except json.JSONDecodeError:
                print(f"   JSON parsing failed: {result[:100]}...")
                return None
                
        except Exception as e:
            print(f"   API call failed: {e}")
            return None
    
    def save_results(self, qa_pairs: List[Dict], output_dir: str = None, ttl_file_path: str = None):
        """
        Save the generated QA pairs to files
        
        Args:
            qa_pairs: List of QA pair dictionaries
            output_dir: Output directory
            ttl_file_path: Original TTL file path to extract base name
        """
        # Use configured output directory if not provided
        if output_dir is None:
            try:
                # Try to load config file to get the correct output directory
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from utils.helpers import load_config

                config = load_config('config.yaml')
                paths = config.get('paths', {})
                output_base_dir = paths.get('output_dir', 'outputs')
                qa_dir = paths.get('qa_dir', 'multi_hop_qa')
                output_dir = str(Path(output_base_dir) / qa_dir)
            except Exception as e:
                # Fallback to outputs/multi_hop_qa if config loading fails
                print(f"Warning: Could not load config file, using default output directory: {e}")
                output_dir = "outputs/multi_hop_qa"

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract base name from TTL file path
        if ttl_file_path:
            base_name = Path(ttl_file_path).stem
            # Remove common suffixes like _extraction, _section_extraction, etc.
            base_name = re.sub(r'_(extraction|section_extraction).*$', '', base_name)
        else:
            base_name = "multi_hop_qa"
        
        # Save detailed results
        detailed_file = output_path / f"{base_name}_qa_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        # Save simplified format for training
        simplified_qa = []
        for qa in qa_pairs:
            simplified_qa.append({
                'section': qa['section'],
                'question': qa['question'],
                'options': qa['options'],
                'correct_answer': qa['correct_answer'],
                'explanation': qa['explanation']
            })
        
        simplified_file = output_path / f"{base_name}_qa_simplified.json"
        with open(simplified_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_qa, f, ensure_ascii=False, indent=2)
        
        # Generate statistics
        stats = self.generate_statistics(qa_pairs)
        stats_file = output_path / f"{base_name}_qa_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats)
        
        print(f"Results saved:")
        print(f"   Detailed version: {detailed_file}")
        print(f"   Simplified version: {simplified_file}")
        print(f"   Statistics: {stats_file}")
    
    def generate_statistics(self, qa_pairs: List[Dict]) -> str:
        """
        Generate statistics about the QA pairs
        
        Args:
            qa_pairs: List of QA pair dictionaries
            
        Returns:
            Statistics string
        """
        stats = []
        stats.append("=== MULTI-HOP QA GENERATION STATISTICS ===\n")
        
        # Overall stats
        stats.append(f"Total QA Pairs Generated: {len(qa_pairs)}")
        
        # By section
        section_counts = {}
        for qa in qa_pairs:
            section = qa['section']
            section_counts[section] = section_counts.get(section, 0) + 1
        
        stats.append(f"\nQA Pairs by Section:")
        for section, count in sorted(section_counts.items()):
            stats.append(f"  {section}: {count}")
        
        # Answer distribution
        answer_dist = {}
        for qa in qa_pairs:
            answer = qa['correct_answer']
            answer_dist[answer] = answer_dist.get(answer, 0) + 1
        
        stats.append(f"\nCorrect Answer Distribution:")
        for answer, count in sorted(answer_dist.items()):
            stats.append(f"  {answer}: {count} ({count/len(qa_pairs)*100:.1f}%)")
        
        # Sample questions
        stats.append(f"\nSample Questions:")
        for i, qa in enumerate(qa_pairs[:3]):
            stats.append(f"\n{i+1}. Section: {qa['section']}")
            stats.append(f"   Question: {qa['question']}")
            stats.append(f"   Correct Answer: {qa['correct_answer']}")
        
        return "\n".join(stats)
    
    def run_pipeline(self, ttl_file_path: str, max_paths_per_section: int = 10, max_qa_per_section: int = 5, output_dir: str = None):
        """
        Run the complete multi-hop QA generation pipeline

        Args:
            ttl_file_path: Path to the TTL knowledge graph file
            max_paths_per_section: Maximum 3-hop paths to find per section
            max_qa_per_section: Maximum QA pairs to generate per section
            output_dir: Output directory for QA files (if None, use config)
        """
        print("Starting multi-hop QA generation pipeline...\n")
        
        # Step 1: Parse TTL file
        print("Step 1: Parsing knowledge graph file...")
        sections = self.parse_ttl_file(ttl_file_path)
        
        # Flatten all triples
        all_triples = []
        for section_triples in sections.values():
            all_triples.extend(section_triples)
        
        print(f"Parsed {len(all_triples)} triples\n")
        
        # Step 2: Find 3-hop paths
        print("Step 2: Finding 3-hop paths...")
        paths = self.find_3hop_paths(all_triples, max_paths_per_section)
        self.hop_paths = paths
        print()
        
        # Step 3: Generate QA pairs
        print("Step 3: Generating QA pairs...")
        qa_pairs = self.generate_qa_pairs(paths, max_qa_per_section)
        self.qa_pairs = qa_pairs
        print()
        
        # Step 4: Save results
        print("Step 4: Saving results...")
        self.save_results(qa_pairs, output_dir=output_dir, ttl_file_path=ttl_file_path)
        
        print("\nMulti-hop QA generation completed!")

def main():
    """
    Main function to run the multi-hop QA generator
    """
    import os
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set the environment variable OPENAI_API_KEY")
        return
    # TODO 要手动测试记得改这里的路径
    # Input file path
    ttl_file = "/home/shenxiaoli/getKG-schema/section_based_extractions/papers_0_merged_split_section_extraction_20250801_165053.ttl"
    
    if not Path(ttl_file).exists():
        print(f"File not found: {ttl_file}")
        return
    
    # Initialize generator
    generator = MultiHopQAGenerator(api_key)
    
    # Run pipeline
    generator.run_pipeline(
        ttl_file_path=ttl_file,
        max_paths_per_section=10,  # 每个章节最多10条3跳路径
        max_qa_per_section=5       # 每个章节最多5个问答对
    )

if __name__ == "__main__":
    main()