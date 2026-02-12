// src/training.rs
use crate::{
    data::{QABatcher, QAProcessor, QABatchItem},
    model::QAModelConfig,
};
use burn::{
    prelude::*,
    optim::{AdamConfig, Optimizer, GradientsParams},
    nn::loss::CrossEntropyLoss,
    tensor::backend::AutodiffBackend,
    record::{BinFileRecorder, FullPrecisionSettings},
};
use burn::data::dataloader::batcher::Batcher;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: QAModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

// The loss function for Q&A is typically Cross-Entropy on start and end token positions.
fn calculate_loss<B: AutodiffBackend>(
    logits: Tensor<B, 3>, // [batch_size, seq_length, 2]
    start_positions: Tensor<B, 1, Int>,
    end_positions: Tensor<B, 1, Int>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [_batch_size, seq_length, _] = logits.dims();

    // Split logits into start and end
    let start_logits: Tensor<B, 2> = logits.clone().slice([0.._batch_size, 0..seq_length, 0..1]).squeeze_dim(2);
    let end_logits: Tensor<B, 2> = logits.slice([0.._batch_size, 0..seq_length, 1..2]).squeeze_dim(2);

    // Calculate cross-entropy loss for both start and end positions
    let loss_start = CrossEntropyLoss::new(None, device).forward(start_logits, start_positions.clone());
    let loss_end = CrossEntropyLoss::new(None, device).forward(end_logits, end_positions.clone());

    // Total loss is the average of the two
    (loss_start + loss_end) / 2.0
}


pub fn run_training<B: AutodiffBackend>(device: B::Device) {
    let config_path = "config.json"; // A file where you store your hyperparameters
    let config: TrainingConfig = serde_json::from_str(
        &std::fs::read_to_string(config_path).expect("Config file not found"),
    ).unwrap();

    // Initialize model
    let mut model: crate::model::QAModel<B> = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    // Load dataset from the 'data' folder
    let dataset = crate::data::load_dataset_from_data_folder("data")
        .expect("Failed to load dataset. Make sure .docx files have corresponding .json files.");
    let processor = QAProcessor::new("data/tokenizer.json", config.model.max_seq_length);
    let tokenized_items: Vec<QABatchItem> = dataset
        .iter()
        .filter_map(|it| processor.process(it))
        .collect();

    let batcher = QABatcher::<B>::new(device.clone());

    // Training loop
    for epoch in 1..=config.num_epochs {
        let mut iter_idx = 0usize;
        for chunk in tokenized_items.chunks(config.batch_size) {
            let batch_items = chunk.iter().cloned().collect::<Vec<_>>();
            let batch = batcher.batch(batch_items, &device);
            let logits = model.forward(batch.tokens, batch.token_type_ids, batch.attention_mask);
            let loss = calculate_loss(logits, batch.start_indices, batch.end_indices, &device);

            // Backpropagation and optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);

            if iter_idx % 10 == 0 {
                println!("Epoch {} | Iter {} | Loss: {:.4}", epoch, iter_idx, loss.into_scalar());
            }
            iter_idx += 1;
        }
        // Save a checkpoint after each epoch
        let model_file = format!("model_epoch_{}", epoch);
        model
            .clone()
            .save_file(&model_file, &BinFileRecorder::<FullPrecisionSettings>::new())
            .expect("Failed to save model");
    }
}
