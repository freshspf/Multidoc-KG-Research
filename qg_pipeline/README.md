# Multi-Agent Knowledge Graph Pipeline

A comprehensive pipeline for extracting, evaluating, and generating QA pairs from academic papers using a multi-agent approach built with LangGraph.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to the project
cd qg_pipeline

# Set up environment variables (copy template first)
cp .env.template .env
# Edit .env to add your OpenAI API key

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Input Data
Place your academic paper JSON files in the `data/processed_data/` directory.

### 3. Run Pipeline
```bash
# Standard processing
python run_pipeline.py

# Large-scale processing (for 1000+ papers)
python run_large_scale.py

# Validate environment only
python run_pipeline.py --validate-env
```

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Dependencies: `openai`, `langgraph`, `pyyaml`

## ⚙️ Configuration

The pipeline uses `config.yaml` for configuration. Key settings:

```yaml
# API Configuration (can use environment variables)
api:
  openai_api_key: ${OPENAI_API_KEY}
  openai_base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}

# Model configurations
extractor:
  model: "gpt-4o-mini"
  temperature: 0.1

evaluator:
  model: "gpt-4o-mini"
  threshold: 6  # Pass/fail threshold (1-10)

qa_generator:
  model: "gpt-4o-mini"
  temperature: 0.7
```

## 📁 Directory Structure

```
qg_pipeline/
├── agents/                    # Multi-agent components
│   ├── extractor/            # Knowledge extraction
│   ├── evaluator/            # Quality evaluation
│   └── QAgenerator/          # QA generation
├── data/                     # Input data
│   └── processed_data/       # Paper JSON files
├── outputs/                  # Generated outputs
│   ├── section_based_extractions/  # Knowledge graphs
│   ├── evaluations/          # Evaluation results
│   ├── multi_hop_qa/         # QA pairs
│   └── logs/                 # Processing logs
├── utils/                    # Utilities
├── config.yaml              # Configuration file
├── .env.template            # Environment variables template
├── run_pipeline.py          # Standard execution
└── run_large_scale.py       # Large-scale execution
```

## 🏗️ Architecture

The pipeline consists of three specialized agents:

1. **SectionBasedExtractor** - Extracts knowledge graphs section by section
2. **TTLEvaluator** - Evaluates knowledge graph quality with GPT-4o
3. **MultiHopQAGenerator** - Generates multi-hop reasoning QA pairs

## 📖 Documentation

For detailed technical documentation, architecture details, and advanced usage, see **[项目报告.md](./项目报告.md)**.

## 🔧 Common Commands

```bash
# Run with custom config
python run_pipeline.py --config my_config.yaml

# Large scale with custom parameters
python run_large_scale.py --batch-size 100 --max-concurrent 2

# Resume from checkpoint
python run_pipeline.py --resume
```

## 📊 Output

- **Knowledge Graphs**: TTL format with section-based extraction
- **Evaluations**: JSON metadata with quality scores
- **QA Pairs**: Multiple-choice questions for multi-hop reasoning

## 🐛 Troubleshooting

1. **API Key Error**: Set `OPENAI_API_KEY` in `.env` file
2. **No Input Files**: Place JSON papers in `data/processed_data/`
3. **Import Errors**: Run `pip install -r requirements.txt`

## 📄 License

Part of the getKG-schema knowledge graph pipeline system.