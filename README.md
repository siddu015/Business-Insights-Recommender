# Business Insights Recommender

Problem Statement: Small businesses struggle to understand evolving markets: buyer behavior, geography-specific shifts, regulatory changes, and competitive pressures. An AI system can help by analyzing multiple data sources, synthesizing insights, and recommending personalized actions.

An AI-powered system that generates personalized business recommendations by fine-tuning FLAN-T5 with LoRA adapters on synthetic business data.

## Overview

This project:

1. Generates synthetic business and market data using LLMs (Gemini)
2. Creates training labels with business recommendations
3. Fine-tunes FLAN-T5 (small) with LoRA for efficient training
4. Generates personalized insights and recommendations for businesses

## Project Structure

```
Business-Insights-Recommender/
├── data/
│   ├── raw/                          # Original synthetic data
│   │   ├── business_data.csv         # 50K business records
│   │   └── market_data.csv           # Market trends by location
│   ├── processed/                    # Engineered features & labels
│   │   ├── engineered_business_data.csv
│   │   ├── generated_templates.json
│   │   └── training_labels.jsonl
│   └── outputs/                      # Inference results
│       └── inference_output.json
├── notebooks/
│   └── analysis_and_engineering.ipynb  # Data exploration & feature engineering
├── scripts/
│   ├── generate_synthetic_data.py    # Creates raw business/market data
│   ├── generate_training_labels.py   # LLM-based label generation
│   ├── train_flan_lora.py           # Fine-tune FLAN-T5 with LoRA
│   └── infer_recommendations.py      # Generate recommendations
├── models/
│   └── flan_t5_lora/                # Trained model & checkpoints
├── logs/
│   └── training_run.log             # Training metrics
├── run_pipeline.py                   # Master orchestrator script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes large model files
└── README.md                         # This file
```

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Business-Insights-Recommender
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey


