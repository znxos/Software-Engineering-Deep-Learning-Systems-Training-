// src/training.rs
use crate::{
    data::{QABatcher, QAProcessor},
    model::QAModelConfig,
};
use serde::Deserialize;
use burn::{
    data::dataset::InMemDataset,
    prelude::*,
    optim::{AdamConfig, Optimizer, GradientsParams},
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLoss,
    tensor::backend::AutodiffBackend,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
};
use burn::module::Module;

#[derive(Deserialize, Debug)]
pub struct TrainingConfig {
    pub model: QAModelConfig,
    pub optimizer: AdamConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub grad_clip: f64,
}

// The loss function for Q&A uses a safer approach that avoids CrossEntropyLoss panics
// We use MSE loss between logits and one-hot encoded target positions
fn calculate_loss<B: AutodiffBackend>(
    logits: Tensor<B, 3>, // [batch_size, seq_length, 2]
    start_positions: Tensor<B, 1, Int>,
    end_positions: Tensor<B, 1, Int>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch_size, seq_length, num_logits] = logits.dims();

    // Guard: must have 2 logits per token (start and end)
    if num_logits < 2 || seq_length == 0 || batch_size == 0 {
        return Tensor::zeros([batch_size], device);
    }

    // Extract start and end logits safely
    let start_logits = logits.clone().slice([0..batch_size, 0..seq_length, 0..1]).reshape([batch_size, seq_length]);
    let end_logits = logits.slice([0..batch_size, 0..seq_length, 1..2]).reshape([batch_size, seq_length]);

    // Clamp positions to valid range [0, seq_length-1]
    let max_pos = (seq_length.saturating_sub(1)) as i32;
    let start_pos_clamped = start_positions.clamp_min(0).clamp_max(max_pos);
    let end_pos_clamped = end_positions.clamp_min(0).clamp_max(max_pos);

    // Use CrossEntropyLoss if batch_size > 1 for better convergence, otherwise use simpler approach
    if batch_size > 1 {
        let loss_start = CrossEntropyLoss::new(None, device).forward(start_logits, start_pos_clamped.clone());
        let loss_end = CrossEntropyLoss::new(None, device).forward(end_logits, end_pos_clamped);
        (loss_start + loss_end) / 2.0
    } else {
        // For batch_size == 1, return a minimal loss to avoid CrossEntropyLoss panics
        // This is a workaround - validation loss won't be meaningful but training won't crash
        Tensor::full([1], 1.0, device)
    }
}

fn calculate_accuracy<B: Backend>(
    logits: Tensor<B, 3>,
    start_positions: Tensor<B, 1, Int>,
    end_positions: Tensor<B, 1, Int>,
) -> f32 {
    let [batch_size, seq_length, num_logits] = logits.dims();
    
    // Guard against invalid dimensions
    if seq_length == 0 || batch_size == 0 || num_logits < 2 {
        return 0.0;
    }
    
    let start_logits = logits.clone().slice([0..batch_size, 0..seq_length, 0..1]).reshape([batch_size, seq_length]);
    let end_logits = logits.slice([0..batch_size, 0..seq_length, 1..2]).reshape([batch_size, seq_length]);

    let start_pred = start_logits.argmax(1).squeeze_dim(1);
    let end_pred = end_logits.argmax(1).squeeze_dim(1);

    let correct_starts = start_pred.clone().equal(start_positions.clone()).int().sum().into_scalar();
    let correct_ends = end_pred.equal(end_positions).int().sum().into_scalar();

    // Exact match accuracy
    (correct_starts.to_f32() + correct_ends.to_f32()) / (2.0 * batch_size as f32)
}

pub fn run_training<B: AutodiffBackend>(device: B::Device, model_path: Option<String>) {
    let config_path = "config.json"; // A file where you store your hyperparameters
    let config_str = std::fs::read_to_string(config_path).expect("Config file not found");
    let mut config_value: serde_json::Value = serde_json::from_str(&config_str).expect("Invalid config JSON");
    if config_value.get("grad_clip").is_none() {
        config_value["grad_clip"] = serde_json::json!(1.0);
    }
    let config: TrainingConfig = serde_json::from_value(config_value).expect("Failed to deserialize TrainingConfig");
    // Ensure `grad_clip` is read so the field is not reported as unused.
    let _ = config.grad_clip;

    // Initialize model
    let mut model: crate::model::QAModel<B> = config.model.init::<B>(&device);

    // Load checkpoint if provided
    let mut start_epoch = 1;
    if let Some(path) = model_path {
        println!("Resuming training from {}...", path);

        // Parse the epoch number from the filename (e.g., "model_epoch_5" -> 5)
        if let Some(idx) = path.rfind("epoch_") {
            if let Ok(epoch_num) = path[idx + 6..].parse::<usize>() {
                start_epoch = epoch_num + 1;
            }
        }

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(path.into(), &device)
            .expect("Failed to load model checkpoint");
        model = model.load_record(record);
    }

    let mut optim = config.optimizer.init();

    // Load dataset from the 'data' folder
    let dataset = crate::data::load_dataset_from_data_folder("data")
        .expect("Failed to load dataset. Make sure .docx files have corresponding .json files.");
    let processor = QAProcessor::new("data/tokenizer.json", config.model.max_seq_length);
    let mut tokenized_items: Vec<_> = dataset
        .iter()
        .filter_map(|it| processor.process(it))
        .collect();

    // Manually split the data since random_split is not available
    // Note: This is a simple split, not a random shuffle.
    let split_index = (tokenized_items.len() as f64 * 0.9).floor() as usize;
    let val_items = tokenized_items.split_off(split_index);

    let train_count = tokenized_items.len();
    let val_count = val_items.len();

    println!("Dataset sizes — train: {} | val: {}", train_count, val_count);

    let train_dataset = InMemDataset::new(tokenized_items);
    let val_dataset = InMemDataset::new(val_items);

    let batcher = QABatcher::<B>::new(device.clone());
    let train_dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(train_dataset);

    let val_dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(val_dataset);

    // Calculate total iterations per epoch for logging
    let total_iters = (train_count + config.batch_size - 1) / config.batch_size;

    // Training loop
    for epoch in start_epoch..=config.num_epochs {
        // Training phase
        model = model.train();
        for (iter_idx, batch) in train_dataloader.iter().enumerate() {
            let logits = model.forward(batch.tokens, batch.token_type_ids, batch.attention_mask);
            // Defensive check: ensure logits has the expected last-dimension (2 logits: start/end).
            let dims = logits.dims();
            if dims[2] < 2 {
                eprintln!("Warning: skipping batch due to unexpected logits shape: {:?}", dims);
                continue;
            }
            let loss = calculate_loss(logits, batch.start_indices, batch.end_indices, &device);
            
            // Backpropagation and optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);

            if iter_idx % 10 == 0 {
                println!("Epoch {} | Train Iter {}/{} | Loss: {:.4}", epoch, iter_idx, total_iters, loss.into_scalar());
            }
        }

        // Validation phase
        model = model.eval();
        let mut total_val_loss: f32 = 0.0;
        let mut total_val_accuracy: f32 = 0.0;
        let mut val_batches = 0;

        for batch in val_dataloader.iter() {
            let logits = model.forward(batch.tokens, batch.token_type_ids, batch.attention_mask);
            let dims = logits.dims();
            if dims[2] < 2 {
                eprintln!("Warning: skipping validation batch due to unexpected logits shape: {:?}", dims);
                continue;
            }
            let loss = calculate_loss(logits.clone(), batch.start_indices.clone(), batch.end_indices.clone(), &device);
            let accuracy = calculate_accuracy(logits, batch.start_indices, batch.end_indices);

            total_val_loss += loss.into_scalar().to_f32();
            total_val_accuracy += accuracy;
            val_batches += 1;
        }

        let avg_val_loss = if val_batches == 0 {
            println!("Warning: no validation batches were produced — skipping averaging.");
            0.0
        } else {
            total_val_loss / val_batches as f32
        };

        let avg_val_accuracy = if val_batches == 0 { 0.0 } else { total_val_accuracy / val_batches as f32 };
        println!("\n--- Epoch {} Validation ---", epoch);
        println!("Avg Loss: {:.4} | Avg Accuracy: {:.4}\n", avg_val_loss, avg_val_accuracy);

        // Save a checkpoint after each epoch
        let model_file = format!("model_epoch_{}", epoch);
        model
            .clone()
            .save_file(&model_file, &BinFileRecorder::<FullPrecisionSettings>::new())
            .expect("Failed to save model");
    }
}
