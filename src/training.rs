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
    tensor::backend::AutodiffBackend,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
};
use burn::module::{Module, AutodiffModule};
use burn::nn::loss::CrossEntropyLossConfig;

#[derive(Deserialize, Debug)]
pub struct TrainingConfig {
    pub model: QAModelConfig,
    pub optimizer: AdamConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
}

/// Compute span extraction loss using L2 regularization on logits
fn compute_span_loss<B: Backend>(
    logits: Tensor<B, 3>,
    start_positions: Tensor<B, 1, Int>,
    end_positions: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let [batch_size, seq_length, _] = logits.dims();
    let device = logits.device();
    
    if batch_size == 0 || seq_length == 0 {
        return Tensor::from_floats([0.1], &device);
    }

    // Split logits into start and end: [batch_size, seq_length]
    let start_logits: Tensor<B, 2> = logits
        .clone()
        .slice([0..batch_size, 0..seq_length, 0..1])
        .squeeze_dim(2);
    let end_logits: Tensor<B, 2> = logits
        .slice([0..batch_size, 0..seq_length, 1..2])
        .squeeze_dim(2);

    let loss_fn = CrossEntropyLossConfig::new()
        .init(&device);

    let start_loss = loss_fn.forward(start_logits, start_positions);
    let end_loss = loss_fn.forward(end_logits, end_positions);

    (start_loss + end_loss) / 2.0
}

fn calculate_accuracy<B: Backend>(
    logits: Tensor<B, 3>,
    start_positions: Tensor<B, 1, Int>,
    end_positions: Tensor<B, 1, Int>,
) -> f32 {
    let [batch_size, seq_length, _] = logits.dims();
    
    if seq_length == 0 || batch_size == 0 {
        return 0.0;
    }
    
    let start_logits: Tensor<B, 2> = logits
        .clone()
        .slice([0..batch_size, 0..seq_length, 0..1])
        .squeeze_dim(2);
    let end_logits: Tensor<B, 2> = logits
        .slice([0..batch_size, 0..seq_length, 1..2])
        .squeeze_dim(2);

    let start_pred = start_logits.argmax(1).squeeze_dim(1);
    let end_pred = end_logits.argmax(1).squeeze_dim(1);

    let correct_starts = start_pred
        .clone()
        .equal(start_positions.clone())
        .int()
        .sum()
        .into_scalar()
        .to_f32();
    let correct_ends = end_pred
        .equal(end_positions)
        .int()
        .sum()
        .into_scalar()
        .to_f32();

    let total = (2.0 * batch_size as f32).max(1.0);
    ((correct_starts + correct_ends) / total).clamp(0.0, 1.0)
}

pub fn run_training<B: AutodiffBackend>(device: B::Device, model_path: Option<String>) {
    let config_path = "config.json";
    let config_str = std::fs::read_to_string(config_path).expect("Config file not found");
    let config: TrainingConfig = serde_json::from_str(&config_str).expect("Failed to deserialize TrainingConfig");

    println!("Training config: learning_rate={}, batch_size={}, num_epochs={}", 
             config.learning_rate, config.batch_size, config.num_epochs);

    // Initialize model
    let mut model: crate::model::QAModel<B> = config.model.init::<B>(&device);

    let mut start_epoch = 1;
    if let Some(path) = model_path {
        println!("Resuming training from {}...", path);
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

    // Load dataset
    let dataset = crate::data::load_dataset_from_data_folder("data")
        .expect("Failed to load dataset");
    let processor = QAProcessor::new("data/tokenizer.json", config.model.max_seq_length);
    let mut tokenized_items: Vec<_> = dataset
        .iter()
        .filter_map(|it| processor.process(it))
        .collect();

    if tokenized_items.is_empty() {
        eprintln!("ERROR: No tokenized items found!");
        return;
    }

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

    let total_iters = (train_count + config.batch_size - 1) / config.batch_size;

    // Training loop
    for epoch in start_epoch..=config.num_epochs {
        model = model.train();
        let mut epoch_loss: f32 = 0.0;
        let mut batch_count: usize = 0;
        let mut consecutive_nans: usize = 0;
        let max_consecutive_nans = 50;

        for (iter_idx, batch) in train_dataloader.iter().enumerate() {
            // Debug first batch
            if iter_idx == 0 && epoch == 1 {
                eprintln!("DEBUG: First batch stats:");
                eprintln!("  Tokens shape: {:?}", batch.tokens.dims());
                eprintln!("  Start indices: {:?}", batch.start_indices.dims());
                eprintln!("  End indices: {:?}", batch.end_indices.dims());
            }

            // Forward pass
            let logits = model.forward(
                batch.tokens.clone(),
                batch.token_type_ids.clone(),
                batch.attention_mask.clone(),
            );

            let dims = logits.dims();
            if dims[0] == 0 || dims[1] == 0 || dims[2] < 2 {
                continue;
            }

            // Debug: Check logits for NaN/Inf before loss computation
            let logits_max = logits.clone().max().into_scalar().to_f32();
            let logits_min = logits.clone().min().into_scalar().to_f32();

            // Compute loss
            let loss = compute_span_loss(
                logits,
                batch.start_indices.clone(),
                batch.end_indices.clone(),
            );

            let loss_val = loss.clone().into_scalar().to_f32();

            if !loss_val.is_finite() {
                consecutive_nans += 1;
                eprintln!("Warning: NaN/inf loss at epoch {} iter {} (reason: bad loss value)", epoch, iter_idx);
                eprintln!("  Logits range: [{:.6}, {:.6}]", logits_min, logits_max);
                eprintln!("  Loss value: {}", loss_val);
                
                if consecutive_nans >= max_consecutive_nans {
                    panic!(
                        "Training diverged: Too many consecutive NaN iterations ({}). Increase learning rate or check data.",
                        consecutive_nans
                    );
                }
                continue;
            }

            consecutive_nans = 0;
            epoch_loss += loss_val;
            batch_count += 1;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // Optimizer step with adaptive learning rate
            let effective_lr = config.learning_rate;
            model = optim.step(effective_lr, model, grads);

            if iter_idx % 10 == 0 {
                let avg_loss = epoch_loss / batch_count.max(1) as f32;
                println!(
                    "Epoch {} | Iter {}/{} | Loss: {:.6} | Avg: {:.6}",
                    epoch, iter_idx, total_iters, loss_val, avg_loss
                );
            }
        }

        let avg_epoch_loss = if batch_count > 0 {
            epoch_loss / batch_count as f32
        } else {
            0.0
        };

        // Validation phase
        let model_valid = model.valid();
        let mut val_loss: f32 = 0.0;
        let mut val_accuracy: f32 = 0.0;
        let mut val_batches: usize = 0;

        for batch in val_dataloader.iter() {
            let tokens = batch.tokens.inner();
            let token_type_ids = batch.token_type_ids.inner();
            let attention_mask = batch.attention_mask.inner();
            let start_indices = batch.start_indices.inner();
            let end_indices = batch.end_indices.inner();

            let logits = model_valid.forward(
                tokens,
                token_type_ids,
                attention_mask,
            );

            let dims = logits.dims();
            if dims[0] == 0 || dims[1] == 0 || dims[2] < 2 {
                continue;
            }

            let loss = compute_span_loss(
                logits.clone(),
                start_indices.clone(),
                end_indices.clone(),
            );
            let loss_f32 = loss.into_scalar().to_f32();
            
            if loss_f32.is_finite() {
                val_loss += loss_f32;
                val_accuracy += calculate_accuracy(
                    logits,
                    start_indices,
                    end_indices,
                );
                val_batches += 1;
            }
        }

        let avg_val_loss = if val_batches > 0 {
            val_loss / val_batches as f32
        } else {
            0.0
        };

        let avg_val_acc = if val_batches > 0 {
            val_accuracy / val_batches as f32
        } else {
            0.0
        };

        println!(
            "\n=== Epoch {} Summary ===\nTrain Loss: {:.6} | Val Loss: {:.6} | Val Acc: {:.4}",
            epoch, avg_epoch_loss, avg_val_loss, avg_val_acc
        );

        // Save checkpoint
        let model_file = format!("model_epoch_{}", epoch);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        model
            .clone()
            .save_file(&model_file, &recorder)
            .expect("Failed to save model checkpoint");
        println!("Saved checkpoint: {}", model_file);
    }

    println!("Training completed!");
}
