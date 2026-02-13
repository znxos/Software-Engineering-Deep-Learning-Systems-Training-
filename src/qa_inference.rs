// src/qa_inference.rs
use crate::{
    data::{extract_text_from_docx, QAProcessor},
    training::TrainingConfig,
};
use burn::{
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor},
};
use std::path::Path;
use anyhow::Result;

pub fn run_inference<B: Backend>(
    doc_path: String,
    question: String,
    model_path: String,
    device: B::Device,
) -> Result<()> {
    // 1. Load configurations from the same file used for training.
    let config_str = std::fs::read_to_string("config.json")?;
    let config: TrainingConfig = serde_json::from_str(&config_str)?;
    let model_config = config.model;

    // NOTE: Ensure you have the tokenizer file at this path.
    // You can download it from the Hugging Face Hub for 'bert-base-uncased'.
    let tokenizer_path = "data/tokenizer.json";
    let processor = QAProcessor::new(tokenizer_path, model_config.max_seq_length);

    // 2. Load the trained model weights.
    println!("Loading model from {}...", &model_path);
    let model = model_config.init::<B>(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(model_path.into(), &device)?;
    let model = model.load_record(record);
    println!("Model loaded successfully.");

    // 3. Load and process the document context.
    println!("Processing document: {}", &doc_path);
    let context = extract_text_from_docx(Path::new(&doc_path))?;

    // 4. Tokenize the context and question for inference.
    let (encoding, tokens, token_type_ids, attention_mask) =
        processor.process_for_inference(&context, &question)
            .ok_or_else(|| anyhow::anyhow!("Failed to tokenize input."))?;

    // Find the start of the context (where token_type_ids are 1) to constrain the search space for the answer.
    let context_start_pos = token_type_ids.iter().position(|&id| id == 1).unwrap_or(0);

    // 5. Convert data into tensors, adding a batch dimension.
    let tokens_vec = tokens.into_iter().map(|t| t as i64).collect::<Vec<i64>>();
    let token_type_vec = token_type_ids.into_iter().map(|t| t as i64).collect::<Vec<i64>>();
    let attention_vec = attention_mask.into_iter().map(|t| t as i64).collect::<Vec<i64>>();

    let tokens_tensor = Tensor::<B, 1, Int>::from_data(tokens_vec.as_slice(), &device).unsqueeze();
    let token_type_ids_tensor = Tensor::<B, 1, Int>::from_data(token_type_vec.as_slice(), &device).unsqueeze();
    let attention_mask_tensor = Tensor::<B, 1, Int>::from_data(attention_vec.as_slice(), &device).unsqueeze();
    let attention_mask_bool = attention_mask_tensor.equal_elem(1);

    // 6. Run the model's forward pass to get logits.
    println!("Running inference...");
    let logits = model.forward(tokens_tensor, token_type_ids_tensor, attention_mask_bool);

    // 7. Find the best start and end token indices from the logits.
    let logits: Tensor<B, 2> = logits.squeeze(); // Remove batch dimension.
    let [seq_length, _] = logits.dims();

    let start_logits_tensor: Tensor<B, 1> = logits.clone().slice([0..seq_length, 0..1]).squeeze();
    let end_logits_tensor: Tensor<B, 1> = logits.slice([0..seq_length, 1..2]).squeeze();

    let start_logits: Vec<f32> = start_logits_tensor.into_data().convert::<f32>().to_vec()?;
    let end_logits: Vec<f32> = end_logits_tensor.into_data().convert::<f32>().to_vec()?;

    let mut best_start_index = 0;
    let mut best_end_index = 0;
    let mut max_score = f32::NEG_INFINITY;

    let max_ans_len = 30; // Maximum length of the answer

    for i in context_start_pos..seq_length {
        for j in i..(i + max_ans_len).min(seq_length) {
            let score = start_logits[i] + end_logits[j];
            if score > max_score {
                max_score = score;
                best_start_index = i;
                best_end_index = j;
            }
        }
    }

    // 8. Decode the answer tokens back to a string and print.
    let all_token_ids = encoding.get_ids();
    if best_start_index <= best_end_index && best_end_index < all_token_ids.len() {
        let answer_tokens = &all_token_ids[best_start_index..=best_end_index];
        let answer = processor.tokenizer.decode(answer_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode answer tokens: {}", e))?;
        println!("\nQuestion: {}\nAnswer: {}", question, answer);
    } else {
        println!("\nQuestion: {}\nAnswer: Could not find a valid answer in the document.", question);
    }

    Ok(())
}