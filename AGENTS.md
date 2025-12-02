# Replay Request
Always respond in Chinese-simplified

# Repository Guidelines

## Project Overview
This is a **multimodal continual learning** research project, focusing on sentiment analysis, aspect term extraction, and named entity recognition tasks with text and images.

### Core Features
- **Multimodal Fusion**: Supports both text-only and text+image modalities
- **Continual Learning**: Supports multiple continual learning strategies (EWC, Replay, LwF, SI, MAS, GEM, etc.)
- **Task Types**: Token-level (MATE, MNER, MABSA) and sentence-level (MASC) tasks
- **Flexible Architecture**: BiLSTM-CRF sequence labeling heads + various continual learning methods

## Project Structure & Config Generation

### Directory Structure
```
MCM/
├── scripts/                    # Training scripts and config generators
│   ├── train.py               # Legacy training script (reference only)
│   ├── train_with_zero_shot.py  # Main training entry (production)
│   ├── generate_*_configs.py  # Config file generators
│   └── configs/               # Generated config files
│       ├── all_task/         # Full task sequence configs
│       ├── quick_test/       # Quick test configs
│       ├── kaggle_ablation/  # Kaggle ablation experiment configs
│       └── kaggle_bilstm_test/ # BiLSTM test configs
├── modules/                   # Core training modules
│   ├── train_refactored.py   # Refactored training logic
│   ├── training_loop_fixed.py # Training loop (with early stopping)
│   ├── evaluate.py           # Evaluation logic
│   ├── parser.py             # Command-line argument parsing
│   └── train_utils.py        # Training utility functions
├── models/                    # Model definitions
│   ├── task_heads/           # Task-specific heads
│   │   ├── mate_head_bilstm.py    # MATE BiLSTM-CRF head
│   │   ├── mner_head_bilstm.py    # MNER BiLSTM-CRF head
│   │   ├── mabsa_head_bilstm.py   # MABSA BiLSTM-CRF head
│   │   ├── masc_head.py           # MASC classification head
│   │   └── get_head.py            # Task head factory function
│   ├── base_model.py         # Base multimodal model
│   └── deqa_expert_model.py  # DEQA expert model
├── datasets/                  # Dataset definitions
│   ├── mate_dataset.py       # MATE dataset
│   ├── mner_dataset.py       # MNER dataset
│   ├── mabsa_dataset.py      # MABSA dataset
│   └── masc_dataset.py       # MASC dataset
├── data/                      # Raw data
│   ├── MASC/                 # MASC MATE MABSA data
│   │   ├── twitter2015       # Twitter2015 text data
│   │   ├── twitter2017       # Twitter2017 text data
│   │   └──mix                # mix text data
│   ├── MNER/twitter2015/     # MNER data
│   └── img/                   # Image data
├── checkpoints/               # Model checkpoints and training info
├── continual/                 # Continual learning components
│   ├── ewc.py                # Elastic Weight Consolidation
│   ├── replay.py             # Experience Replay
│   ├── lwf.py                # Learning without Forgetting
│   └── metrics.py            # Continual learning metrics
├── utils/                     # Utility functions
│   ├── plot.py               # Visualization tools (heatmaps, training curves)
│   ├── metrics_utils.py      # Metrics computation
│   └── logger.py             # Logging tools
├── visualize/                 # Visualization modules
│   └── training_curves.py    # Training curve plotting
└── tests/                     # Test files
    ├── test_crf_pipline.py   # CRF pipeline tests
    └── simple_ner_training.py # Simple NER training example
```

### Config File Generation
Use `scripts/generate_*_configs.py` to generate complete continual learning configs:
- `generate_task_config.py`: Generate full task sequence configs
- `generate_quick_test_config.py`: Generate quick test configs (200 samples, 2 epochs)
- `generate_kaggle_ablation_configs.py`: Generate Kaggle ablation experiment configs
- `generate_kaggle_bilstm_test_configs.py`: Generate BiLSTM test configs

**Config File Format**:
```json
{
  "tasks": [
    {
      "session_name": "mate_twitter2015_textonly",
      "task_name": "mate",
      "train_text_file": "data/MASC/twitter2015/train.txt",
      "epochs": 15,
      "batch_size": 16,
      "use_bilstm": 0,
      "bilstm_hidden_size": 256,
      ...
    },
    ...
  ],
  "global_params": {
    "train_info_json": "checkpoints/train_info.json",
    ...
  }
}
```

## Tasks, Datasets & Baselines

### Datasets
- **Twitter2015**: 2015 Twitter data with text and images
- **Twitter2017**: 2017 Twitter data with text and images
- **Mixed Dataset**: Twitter2015 + Twitter2017

### Task Types

#### Token-level Tasks (Sequence Labeling)
1. **MATE (Multi-modal Aspect Term Extraction)**: Aspect term extraction
   - Labels: 3 (O, B-ASPECT, I-ASPECT)
   - Primary Metric: Chunk F1 (Span-level Micro F1)

2. **MNER (Multi-modal Named Entity Recognition)**: Named entity recognition
   - Labels: 9 (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC)
   - Primary Metric: Chunk F1 (Span-level Micro F1)

3. **MABSA (Multi-modal Aspect-Based Sentiment Analysis)**: Aspect-level sentiment analysis
   - Labels: 7 (O, B-POS, I-POS, B-NEU, I-NEU, B-NEG, I-NEG)
   - Primary Metric: Chunk F1 (Span-level Micro F1)

#### Sentence-level Tasks
4. **MASC (Multi-modal Aspect-based Sentiment Classification)**: Sentence-level sentiment classification
   - Labels: 3 (Positive, Neutral, Negative)
   - Primary Metric: Micro F1

### Modality Types
- **text_only**: Text only
- **multimodal**: Text + Image (CLIP image encoding)

### Continual Learning Strategies (Baselines)
- **finetune**: Simple fine-tuning (no continual learning)
- **ewc**: Elastic Weight Consolidation (10.1073/pnas.1611835114)
- **replay**: Experience Replay
- **lwf**: Learning without Forgetting (10.1007/978-3-319-46493-0_37)
- **si**: Synaptic Intelligence (10.48550/arXiv.1703.04200)
- **mas**: Memory Aware Synapses (10.48550/arXiv.1711.09601)
- **gem**: Gradient Episodic Memory (10.48550/arXiv.1706.08840)
- **tam_cl**: Task-Adaptive Memory (10.48550/arXiv.2303.14423)
- **moeadapter**: Mixture-of-Experts Adapters (10.48550/arXiv.2403.11549)
- **clap4clip**: CLAP for CLIP (10.48550/arXiv.2403.19137)

### Task Sequences
1. **Sequence 1**: `asc→ate→ner→absa→masc→mate→mner→mabsa`
2. **Sequence 2**: `ate→ner→absa→asc→mate→mner→mabsa→masc`

## Launch Modes & Environments

### 1. Local Quick Test
**Purpose**: Ensure code runs, quick validation with 200 samples
```bash
# Generate quick test config
python scripts/generate_quick_test_config.py

# Run test
python -m scripts.train_with_zero_shot --config scripts/configs/quick_test/quick_test_config.json
```

### 2. Manual Run
**Purpose**: Development and debugging
```bash
python -m scripts.train_with_zero_shot --config scripts/configs/<flow>.json
```

### 3. Shared Server (SSH Offline)
**Purpose**: Multi-user shared GPU server
```bash
# Use GPU wait script
cd scripts/configs/all_task
./start_all_experiments.sh  # Wait for GPU idle then auto-run
```

### 4. AutoDL Rental Server
**Purpose**: Hourly-billed cloud GPU
- **Note**: Put data in fast I/O directory, close instance promptly
- **Reference**: `scripts/configs/autodl_config/run_autodl_experiments_fixed.sh`
- **Features**: Includes email reminders and auto-shutdown

### 5. Kaggle Free GPU
**Purpose**: Leverage Kaggle free GPU resources (12-hour limit)
- **Reference**: `scripts/configs/kaggle_ablation/` and `scripts/configs/kaggle_bilstm_test/`
- **Limitation**: Each run <9-12 hours, need to split experiments
- **Workflow**:
  1. Upload project as Kaggle dataset
  2. Create Notebook, add dataset
  3. Run `run_account_X.py`
  4. Export results to `/kaggle/working/`

## Build, Test, and Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Config Generation
```bash
# Generate full task config
python scripts/generate_task_config.py

# Generate quick test config
python scripts/generate_quick_test_config.py

# Generate Kaggle BiLSTM test configs
python scripts/generate_kaggle_bilstm_test_configs.py --account all
```

### Training Commands
```bash
# Basic training
python -m scripts.train_with_zero_shot --config scripts/configs/session.json

# Quick test
python -m scripts.train_with_zero_shot --config scripts/configs/quick_test/quick_test_config.json
```

### Test Commands
```bash
# Run specific test
pytest tests/test_crf_pipline.py

# Run all tests
pytest tests/

# Run specific module tests
pytest tests/ -k crf
pytest tests/ -k deqa
```

### Result Analysis
```bash
# View training info
cat checkpoints/train_info_*.json

# Generate heatmaps (auto-generated after training)
# Heatmaps saved in checkpoints/*.png
```

## Coding Style & Naming Conventions
- **Python Version**: 3.8+
- **Indentation**: 4 spaces
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Comments**: English preferred, comment critical logic
- **Script Naming**: `train_<task>_<dataset>_<strategy>.sh`
- **Logging**: Use `utils.logger.setup_logger`
- **Device Detection**: Explicitly use `torch.cuda.is_available()`
- **String Formatting**: Prefer f-strings


## Evaluation Metrics

### Token-level Task Metrics
- **Chunk F1 (Primary)**: Span-level Micro F1, evaluates entity boundaries and categories
- **Token Micro F1 (no O)**: Token-level F1, excluding O label
- **Token Macro F1**: Token-level macro-averaged F1
- **Token Accuracy**: Token-level accuracy (reference only)

### Sentence-level Task Metrics
- **Micro F1 (Primary)**: Micro-averaged F1
- **Macro F1**: Macro-averaged F1
- **Accuracy**: Accuracy

### Continual Learning Metrics
- **AA (Average Accuracy)**: Average accuracy
- **AIA (Average Incremental Accuracy)**: Average incremental accuracy
- **FM (Final Model)**: Final model's average performance across all tasks
- **BWT (Backward Transfer)**: Backward transfer (forgetting degree)

### Training Monitoring
- **Best Dev Metric**: Best dev metric (for early stopping)
- **Best Epoch**: Epoch with best dev metric
- **Dev Loss**: Validation loss (for training curves)

## Visualization & Analysis

### Auto-generated Visualizations
1. **Heatmaps**: Continual learning accuracy matrices
   - `accuracy_heatmap.png`: Default metric (acc)
   - `chunk_f1_heatmap.png`: Chunk F1 metric
   - `token_micro_f1_heatmap.png`: Token Micro F1 metric

2. **Training Curves**:
   - Train Loss vs Epochs
   - Dev Loss vs Epochs
   - Span F1 vs Epochs

### Manual Analysis Scripts
```bash
# Analyze BiLSTM test results
python scripts/configs/kaggle_bilstm_test/analyze_bilstm_results.py <results_dir>
```

## Commit & Pull Request Guidelines
- **Commit Format**: Concise imperatives (e.g., `Add BiLSTM support`, `Fix CRF mask bug`)
- **Description**: Reference affected modules (e.g., "touches `continual/label_embedding.py`")
- **Links**: Associate related issues or experiment IDs
- **Commands**: Include complete run commands
- **Metrics**: Attach key metrics or charts
- **Data**: Explain data/checkpoint updates, list configs needed for reproduction

### Recent Important Commits
- `BiLSTM-CRF integration`: Integrated BiLSTM-CRF sequence labeling heads
- `Add best dev metric tracking`: Track best dev metrics and epoch
- `Fix training curves visualization`: Fixed training curve plotting
- `Kaggle BiLSTM test configs`: Added Kaggle BiLSTM test configs

## Security & Configuration Tips
- **Paths**: Use paths relative to repo root
- **Credentials**: Do not commit to Git
- **Checkpoints**: Keep large files outside Git
- **Directories**: Call `ensure_directory_exists` before writing files
- **Sandbox**: Keep script runs sandbox-safe

## Key Features & Recent Updates



### Training Improvements
- **Early Stopping**: Stop early based on dev metrics
- **Best Metric Tracking**: Track best dev metrics and corresponding epoch
- **Dev Loss Recording**: Record dev loss for each epoch
- **Training Summary**: Print detailed summary after training

### Visualization Enhancements
- **Training Curves**: Auto-plot loss and F1 curves
- **Multiple Heatmaps**: Support three types of metric heatmaps
- **Metric-specific**: Display correct metric names based on task type

### Kaggle Support
- **Multi-account**: Support parallel testing with multiple accounts
- **Time Management**: Split experiments considering 12-hour limit
- **Result Export**: Auto-collect and export results

