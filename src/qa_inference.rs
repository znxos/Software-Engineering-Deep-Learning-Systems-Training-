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
    let mut config_value: serde_json::Value = serde_json::from_str(&config_str)?;
    if config_value.get("grad_clip").is_none() {
        config_value["grad_clip"] = serde_json::json!(1.0);
    }
    let config: TrainingConfig = serde_json::from_value(config_value)?;
    let model_config = config.model;

    // NOTE: Ensure you have the tokenizer file at `data/tokenizer.json`.
    // You can download it from the Hugging Face Hub for 'bert-base-uncased'.
    let processor = QAProcessor::new("data/tokenizer.json", model_config.max_seq_length);

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

    // 4. Sliding Window Inference
    // We tokenize the full context and question, then process in overlapping windows.
    let q_encoding = processor.tokenizer.encode(question.clone(), false).map_err(|e| anyhow::anyhow!(e))?;
    let c_encoding = processor.tokenizer.encode(context, false).map_err(|e| anyhow::anyhow!(e))?;
    
    let q_ids: Vec<u32> = q_encoding.get_ids().iter().map(|&x| x as u32).collect();
    let c_ids: Vec<u32> = c_encoding.get_ids().iter().map(|&x| x as u32).collect();

    let max_len = model_config.max_seq_length;
    // Calculate available space for context: max_len - question_len - 3 special tokens ([CLS], [SEP], [SEP])
    let available_ctx_len = max_len.saturating_sub(q_ids.len() + 3);
    
    if available_ctx_len == 0 {
        return Err(anyhow::anyhow!("Question is too long for the model's max sequence length."));
    }

    // Stride determines how much we move the window. 80% of available space is a good balance.
    let stride = (available_ctx_len as f32 * 0.8) as usize; 

    let mut best_answer = String::from("No answer found");
    let mut best_score = f32::NEG_INFINITY;
    let mut found_valid_answer = false;

    println!("Running inference with sliding window (Stride: {})...", stride);

    let mut window_start = 0;
    while window_start < c_ids.len() {
        let window_end = (window_start + available_ctx_len).min(c_ids.len());
        let chunk = &c_ids[window_start..window_end];

        // Construct input: [CLS] Question [SEP] Context_Chunk [SEP]
        let mut tokens = Vec::new();
        tokens.push(101); // CLS
        tokens.extend_from_slice(&q_ids);
        tokens.push(102); // SEP
        let context_start_offset = tokens.len();
        tokens.extend_from_slice(chunk);
        tokens.push(102); // SEP

        // Pad to max_len
        let seq_len = tokens.len();
        let pad_len = max_len.saturating_sub(seq_len);
        tokens.extend(vec![0; pad_len]);

        // Token Types: 0 for Question, 1 for Context
        let mut token_type_ids = vec![0; context_start_offset];
        token_type_ids.extend(vec![1; chunk.len() + 1]); // +1 for trailing SEP
        token_type_ids.extend(vec![0; pad_len]);

        // Attention Mask
        let mut attention_mask = vec![1; seq_len];
        attention_mask.extend(vec![0; pad_len]);

        // Convert to Tensors
        let tokens_tensor = Tensor::<B, 1, Int>::from_data(
            tokens.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(), 
            &device
        ).unsqueeze(); // [1, seq_len]
        
        let token_type_ids_tensor = Tensor::<B, 1, Int>::from_data(
            token_type_ids.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(), 
            &device
        ).unsqueeze();

        let attention_mask_tensor = Tensor::<B, 1, Int>::from_data(
            attention_mask.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(), 
            &device
        ).unsqueeze();
        let attention_mask_bool = attention_mask_tensor.equal_elem(1);

        // Forward pass
        let logits = model.forward(tokens_tensor, token_type_ids_tensor, attention_mask_bool);
        let logits: Tensor<B, 2> = logits.squeeze_dim(0); // [seq_len, 2]

        let start_logits: Tensor<B, 1> = logits.clone().slice([0..seq_len, 0..1]).squeeze_dim(1);
        let end_logits: Tensor<B, 1> = logits.slice([0..seq_len, 1..2]).squeeze_dim(1);

        let start_probs: Vec<f32> = start_logits.into_data().convert::<f32>().to_vec()?;
        let end_probs: Vec<f32> = end_logits.into_data().convert::<f32>().to_vec()?;

        // Search for best span in this window (only within the context part)
        let context_end_offset = context_start_offset + chunk.len();
        
        for i in context_start_offset..context_end_offset {
            for j in i..(i + 30).min(context_end_offset) { // Max answer length 30 tokens
                let score = start_probs[i] + end_probs[j];
                if score > best_score {
                    best_score = score;
                    let answer_tokens = &tokens[i..=j];
                    if let Ok(decoded) = processor.tokenizer.decode(answer_tokens, true) {
                        best_answer = decoded;
                        found_valid_answer = true;
                    }
                }
            }
        }

        if window_end == c_ids.len() { break; }
        window_start += stride;
    }

    // Clean up the answer
    if found_valid_answer {
        let clean_answer = best_answer
            .replace("[CLS]", "")
            .replace("[SEP]", "")
            .replace("[PAD]", "")
            .trim()
            .to_string();
        
        println!("\nQuestion: {}\nAnswer: {}", question, clean_answer);
        println!("Confidence Score: {:.4}", best_score);
    } else {
        println!("\nQuestion: {}\nAnswer: No valid answer found.", question);
    }

    Ok(())
}