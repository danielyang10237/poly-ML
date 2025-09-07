# ML Classification Projects

This repository contains several Natural Language Processing (NLP) projects focused on text analysis, sentiment classification, and performance categorization. The projects demonstrate various machine learning approaches including transformer models, BiLSTM networks, and large language model integration.

## Overview of All Projects

### 1. Clause Segmentation
**Location**: `Clause Segmentation/`

Splits complex sentences into individual clauses using spaCy's transformer-based English model (`en_core_web_trf`). The system identifies sentence roots, subjects, and dependencies to extract meaningful clause segments.

**Key Features**:
- Uses spaCy's transformer model for accurate dependency parsing
- Identifies sentence roots and subjects
- Extracts clauses based on grammatical relationships
- Handles complex sentences with multiple subjects and verbs

**Files**:
- `parsing_script.py` - Main clause segmentation implementation
- `spacy_split.ipynb` - Jupyter notebook for experimentation

### 2. Sentiment Analysis
**Location**: `Sentiment/`

Two different approaches to sentiment analysis:

#### BiLSTM Implementation (`BiLSTM.ipynb`)
- Custom BiLSTM neural network for sentiment and sarcasm detection
- Uses BERT tokenization for preprocessing
- Multi-task learning: sentiment (positive/negative/neutral) and sarcasm detection
- Trained on Reddit, Twitter, and sarcasm headline datasets

#### Transfer Learning Approach (`pure_sentiment.ipynb`)
- Fine-tuned RoBERTa model (`siebert/sentiment-roberta-large-english`)
- Binary sentiment classification (positive/negative)
- Achieves 96.6% accuracy on validation set

**Datasets**:
- Reddit and Twitter sentiment data
- Sarcasm headlines dataset (v1 and v2)
- Amazon product reviews

### 3. Performance Classification
**Location**: `Performance/`

Multi-label classification system for categorizing product reviews into performance areas using DeBERTa-v3-large model.

**Categories**:
- Customer Service, Audio, Visual, Bluetooth, Microphone
- Battery Life, Internet, Performance/Speed, Value/Price
- Ease of Use, Comfortability/Fit, Undetermined

**Key Features**:
- Uses OpenAI GPT-3.5-turbo for dataset labeling
- DeBERTa-v3-large fine-tuned for multi-label classification
- Includes data collection and preprocessing scripts
- Checkpoint system for resumable processing

**Files**:
- `collect_dataset.py` - Data collection with OpenAI API
- `create_dataset_budget.py` - Batch processing with checkpointing
- `update_dataset.py` - Dataset management utilities
- `model_epoch_2.pth` - Trained model checkpoint

### 4. Insights (LLM Integration)
**Location**: `Insights/`

Simple integration with Ollama's Llama 3.1 model for general language model tasks.

**Features**:
- Uses LangChain for Ollama integration
- Local LLM deployment with Llama 3.1
- Basic text generation capabilities

### 5. Demo Application
**Location**: `demo.py`

Interactive GUI application combining sentiment analysis and performance classification.

**Features**:
- Tkinter-based graphical interface
- Real-time sentiment and performance classification
- Futuristic dark theme design
- Input validation and error handling

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- MySQL database (for database integration features)

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate NLP_env3.8
```

### Additional Setup

#### 1. Download spaCy Models
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_trf
```

#### 2. Install Ollama (for Insights project)
```bash
# Run the installation script
chmod +x install.sh
./install.sh

# Pull the Llama model
ollama pull llama3.1
```

#### 3. Environment Variables
Create a `.env` file for API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Key Dependencies

### Core ML Libraries
- **PyTorch** (2.2.2) - Deep learning framework
- **Transformers** (4.43.3) - Hugging Face transformers library
- **spaCy** (3.7.2) - NLP processing
- **scikit-learn** (1.3.0) - Machine learning utilities

### NLP Models
- **BERT** - Tokenization and embeddings
- **RoBERTa** - Sentiment analysis
- **DeBERTa-v3-large** - Performance classification
- **spaCy transformer models** - Clause segmentation

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **gensim** - Topic modeling and text processing
- **NLTK** - Natural language toolkit

### GUI and Utilities
- **tkinter** - GUI framework
- **requests** - HTTP client
- **mysql-connector-python** - Database connectivity
- **langchain-ollama** - LLM integration

## Project Structure

```
ML_Models/
├── Clause Segmentation/          # Sentence clause extraction
├── Sentiment/                   # Sentiment analysis models
├── Performance/                 # Performance classification
├── Insights/                    # LLM integration
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── demo.py                      # Interactive demo application
├── label_database.py           # Database integration
└── install.sh                  # Ollama installation script
```
