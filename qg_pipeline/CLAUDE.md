# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent knowledge graph pipeline that extracts, evaluates, and generates QA pairs from academic papers using LangGraph. The system processes academic papers through three specialized agents to create high-quality knowledge graphs and multi-hop reasoning questions.

## Core Commands

### Environment Setup
```bash
# Set required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional

# Install dependencies
pip install -r requirements.txt

# Validate environment
python run_pipeline.py --validate-env
```

### Running the Pipeline
```bash
# Basic pipeline execution
python run_pipeline.py

# With custom configuration
python run_pipeline.py --config my_config.yaml

# Large scale processing (for 1000+ papers)
python run_large_scale.py

# Resume from checkpoint
python run_pipeline.py --resume
```

### Testing and Validation
```bash
# Verify installation
python -c "import openai, langgraph, torch; print('✅ All packages installed successfully')"

# Check log files
tail -f outputs/logs/pipeline_*.log
```

## Architecture

### Multi-Agent System
The pipeline uses LangGraph to coordinate three specialized agents:

1. **SectionBasedExtractor** (`agents/extractor/`)
   - Extracts knowledge graphs from academic papers section by section
   - Outputs TTL/RDF format files
   - Handles formulas, model architectures, and examples

2. **TTLEvaluator** (`agents/evaluator/`)
   - Evaluates knowledge graph quality using GPT-4o
   - Provides scores across multiple dimensions
   - Generates improvement suggestions for failed evaluations

3. **MultiHopQAGenerator** (`agents/QAgenerator/`)
   - Generates multi-hop reasoning questions from high-quality KGs
   - Creates multiple-choice QA pairs
   - Only processes KGs that pass evaluation threshold

### Workflow Coordination
- **LangGraph StateGraph**: Manages agent coordination and conditional routing
- **State Management**: Persistent state tracking via `utils/storage.py`
- **Retry Logic**: Automatic retry with improvement suggestions for failed evaluations
- **Batch Processing**: Sequential processing with progress tracking

### Configuration System
The pipeline uses a comprehensive configuration system with YAML-based configuration files and environment variable support:

#### Configuration Architecture
- **`config.yaml`**: Main configuration file with all system parameters
- **`.env`**: Environment variables for API keys and runtime settings
- **`.env.template`**: Template file with all configurable variables documented

#### Key Configuration Sections
- **`api`**: OpenAI API settings (keys, base URLs)
- **`paths`**: Input/output directory configurations
- **`processing`**: Batch processing and retry parameters
- **`extractor`**: Knowledge extraction model and parameters
- **`evaluator`**: Evaluation model, threshold, and scoring settings
- **`qa_generator`**: QA generation model and creativity parameters
- **`workflow`**: Processing flow and logging configurations
- **`retry`**: Retry logic and improvement suggestion settings
- **`large_scale`**: Settings for processing 1000+ papers with parallel batches

#### Model Configuration
All agents support configurable LLM models:
```yaml
extractor:
  model: "deepseek-v3.1"      # Extraction model
  temperature: 0.1            # Generation temperature
  max_tokens: 3000           # Token limit

evaluator:
  model: "deepseek-v3.1"      # Evaluation model
  threshold: 9               # Pass/fail threshold

qa_generator:
  model: "deepseek-v3.1"      # QA generation model
  temperature: 0.7           # Creativity level
```

## Key File Locations

### Entry Points
- `run_pipeline.py`: Standard pipeline execution
- `run_large_scale.py`: Large-scale processing for 1000+ papers
- `workflow.py`: Core LangGraph workflow implementation
- `large_scale_workflow.py`: Optimized workflow for large datasets

### Input/Output
- `data/`: Input paper JSON files
- `outputs/section_based_extractions/`: Generated TTL files and metadata
- `outputs/multi_hop_qa/`: Generated QA pairs (detailed/simplified/stats)
- `outputs/results/`: Batch processing reports
- `outputs/state/`: Workflow state and checkpoints
- `outputs/logs/`: Processing logs

### Utilities
- `utils/helpers.py`: Configuration loading, logging, file utilities
- `utils/storage.py`: State management, batch tracking, results storage

## Development Guidelines

### Agent Development
Each agent follows a consistent pattern:
- Initialize with API credentials and configuration
- Implement main processing method that accepts input and returns structured output
- Handle errors gracefully with detailed logging
- Support configuration parameters for model, temperature, max tokens

### State Management
Use the `StateManager` and `BatchState` classes for:
- Tracking processing progress across papers
- Persisting intermediate results
- Enabling resume functionality
- Managing retry attempts and suggestions

### Configuration Management
When working with configuration files:

#### Environment Setup
```bash
# Create .env file from template
cp .env.template .env
# Edit with your actual API keys
nano .env

# Load environment variables
source .env
```

#### Model Configuration
- All agents read model settings from `config.yaml`
- Models can be switched without code changes
- Support for different models per agent (extraction, evaluation, QA generation)

#### Key Configuration Guidelines
- **Evaluation threshold**: Lower values (6-7) allow more papers to proceed to QA generation
- **Batch size**: Keep at 1 for stability unless using large-scale mode
- **Temperature settings**: Lower for extraction/evaluation (0.1), higher for QA generation (0.7)
- **Token limits**: Adjust based on model capabilities and cost considerations

#### Configuration Validation
```bash
# Validate environment and configuration
python run_pipeline.py --validate-env

# Test configuration loading
python -c "from utils.helpers import load_config; print(load_config('config.yaml'))"
```

### Error Handling
The pipeline includes comprehensive error handling:
- API failures trigger automatic retries with exponential backoff
- Evaluation failures provide specific improvement suggestions
- Processing errors are logged with full tracebacks
- State is preserved to enable resumption after failures