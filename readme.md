# Word Document Q&A System

A comprehensive Question-Answering system built in **Rust** using the **Burn** deep learning framework. This system reads Word documents (.docx) and answers natural language questions about their content using transformer-based neural networks with span extraction.

## Example Questions

- "What is the Month and date of the 2026 End of Year Graduation Ceremony?"
- "How many times did the HDC hold their meetings in 2024?"

---

## Prerequisites

### 1. Install Rust

Visit [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install) and follow the installation guide for your operating system.

### 2. Install Visual Studio Build Tools (Windows)

This project requires C++ build tools:

1. Download the [Visual Studio Installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer
3. Select **"Desktop Development with C++"**
4. Complete the installation

---

## Data Requirements

Before training or running inference, ensure the following files exist in the `data/` directory:

### Required Document Files
- `calendar_2024.docx` – Calendar document for 2024
- `calendar_2025.docx` – Calendar document for 2025
- `calendar_2026.docx` – Calendar document for 2026

### Required Data Pairs
- `calendar_2024.json` – Q&A pairs for 2024 document
- `calendar_2025.json` – Q&A pairs for 2025 document
- `calendar_2026.json` – Q&A pairs for 2026 document

### Tokenizer
- `data/tokenizer.json` – BERT base-uncased tokenizer (already included)

### Optional
- `augment_training_data.py` – Script to generate additional training data variations

---

## Build

Compile the project in release mode with optimizations:

```bash
cargo build --release
```

The compiled binary will be available in `target/release/word-doc-qa`.

---

## Training

### Train from Scratch

Start training the model from the beginning:

```bash
cargo run --release -- train
```

The system will:
- Load calendar documents and Q&A data from `data/`
- Initialize the transformer model
- Begin training for the configured number of epochs
- Save checkpoints as `model_epoch_0`, `model_epoch_1`, etc.

### Resume Training

Resume training from a saved checkpoint:

```bash
cargo run --release -- train --model-path "model_epoch_N"
```

**Replace `N` with the epoch number** you want to resume from. For example:

```bash
# Resume from epoch 5
cargo run --release -- train --model-path "model_epoch_5"

# Resume from epoch 28
cargo run --release -- train --model-path "model_epoch_28"
```

The model will load from the checkpoint and continue training, resuming at the next epoch.

---

## Inference

Run interactive Q&A inference on a document using a trained model:

### Example 1: Document Specific
```bash
cargo run --release -- infer --doc-path <path_to_document> --model-path <checkpoint_path>
```
```bash
cargo run --release -- infer --doc-path data/calendar_2026.docx --model-path model_epoch_28
```

This will:
- Load the specified document
- Load the trained model checkpoint
- Start an interactive loop where you can enter questions
- Display predicted answers extracted from the document

### Pre-trained Model

Pre-trained model checkpoints are available for download:

📥 **[Download trained model](https://mega.nz/folder/sKAwARbB#zRwOoIkNyP04QdmJd1KGbA)**

Save the downloaded models to your project root directory, then use them for inference without retraining.
---

## Hardware & GPU Support

### GPU Acceleration

This project uses **WGPU** for GPU acceleration, enabling efficient inference and training:

- **Windows**: Uses DirectX 12 (DX12) by default
- **Linux**: Uses Vulkan
- **macOS**: Uses Metal

## Model Architecture

- **Type**: Transformer-based span extractor
- **Layers**: 6 transformer encoder layers
- **Attention Heads**: 8
- **Model Dimension**: 512
- **Vocabulary Size**: 30,522 (BERT base-uncased)
- **Max Sequence Length**: 512 tokens
- **Output**: Start and end position logits for answer span

---

## Configuration

Training parameters are specified in `config.json`:

- `num_epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Initial learning rate (default: 0.00002)
- `model`: Model architecture hyperparameters

---

## Project Structure

```
.
├── src/
│   ├── main.rs           # CLI entry point
│   ├── model.rs          # Neural network architecture
│   ├── data.rs           # Data loading and preprocessing
│   ├── training.rs       # Training loop
│   └── qa_inference.rs   # Inference pipeline
├── data/
│   ├── calendar_*.docx   # Input documents
│   ├── calendar_*.json   # Q&A training data
│   └── tokenizer.json    # BERT tokenizer
├── config.json           # Training configuration
├── Cargo.toml            # Project manifest
└── README.md             # This file
```

---

## Dependencies

- **Burn 0.20.1** – Deep learning framework
- **docx-rs 0.4** – DOCX document parsing
- **tokenizers 0.15** – BERT tokenization
- **serde** – Serialization
- **clap** – Command-line argument parsing

For complete dependencies, see `Cargo.toml`.

---

## Development

### Full Build with Tests

```bash
cargo build --release
```

---

## Troubleshooting

### Common Issues

**"Failed to create GPU device"**
- Ensure graphics drivers are up-to-date
- On Windows, ensure Visual C++ Build Tools are installed

**Build fails with missing dependencies**
- Run `cargo clean` and rebuild: `cargo build --release`
- Ensure Rust is up-to-date: `rustup update`

**Out of memory during training**
- Reduce `batch_size` in `config.json`
- Use a smaller model checkpoint for inference

---

## Notes

- **Do not change dependency versions** in `Cargo.toml`. The project is calibrated for the specified versions.

- The system includes robust error handling and NaN detection during training.

- Model checkpoints are automatically saved after each epoch.

---

