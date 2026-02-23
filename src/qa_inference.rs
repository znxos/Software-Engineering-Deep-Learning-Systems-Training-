// src/qa_inference.rs
use crate::{
    data::{extract_text_from_docx, QAProcessor},
    training::TrainingConfig,
};
use burn::{
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::activation::softmax,
    tensor::{Int, Tensor},
};
use std::path::Path;
use anyhow::Result;
use std::io::{self, Write};

pub fn run_inference<B: Backend>(
    doc_path: String,
    _question: String, // Not used interactively
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

    println!("\nEnter your questions (type 'exit' to quit):");
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut question = String::new();
        io::stdin().read_line(&mut question)?;
        let question = question.trim();
        if question.eq_ignore_ascii_case("exit") {
            break;
        }
        if question.is_empty() {
            continue;
        }

        // --- Inference logic ---
        let q_encoding = processor.tokenizer.encode(question.to_string(), false).map_err(|e| anyhow::anyhow!(e))?;
        let c_encoding = processor.tokenizer.encode(context.clone(), false).map_err(|e| anyhow::anyhow!(e))?;
        let q_ids: Vec<u32> = q_encoding.get_ids().iter().map(|&x| x as u32).collect();
        let c_ids: Vec<u32> = c_encoding.get_ids().iter().map(|&x| x as u32).collect();
        let max_len = model_config.max_seq_length;
        let available_ctx_len = max_len.saturating_sub(q_ids.len() + 3);
        if available_ctx_len == 0 {
            println!("Question is too long for the model's max sequence length.");
            continue;
        }
        let stride = (available_ctx_len as f32 * 0.8) as usize;
        let mut best_answer = String::from("No answer found");
        let mut best_score = f32::NEG_INFINITY;
        let mut found_valid_answer = false;
        let mut window_start = 0;
        while window_start < c_ids.len() {
            let window_end = (window_start + available_ctx_len).min(c_ids.len());
            let chunk = &c_ids[window_start..window_end];
            let mut tokens = Vec::new();
            tokens.push(101); // CLS
            tokens.extend_from_slice(&q_ids);
            tokens.push(102); // SEP
            let context_start_offset = tokens.len();
            tokens.extend_from_slice(chunk);
            tokens.push(102); // SEP
            let seq_len = tokens.len();
            let pad_len = max_len.saturating_sub(seq_len);
            tokens.extend(vec![0; pad_len]);
            let mut token_type_ids = vec![0; context_start_offset];
            token_type_ids.extend(vec![1; chunk.len() + 1]);
            token_type_ids.extend(vec![0; pad_len]);
            let mut attention_mask = vec![1; seq_len];
            attention_mask.extend(vec![0; pad_len]);
            let tokens_tensor = Tensor::<B, 1, Int>::from_data(
                tokens.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(), 
                &device
            ).unsqueeze();
            let token_type_ids_tensor = Tensor::<B, 1, Int>::from_data(
                token_type_ids.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(), 
                &device
            ).unsqueeze();
            let attention_mask_tensor = Tensor::<B, 1, Int>::from_data(
                attention_mask.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(), 
                &device
            ).unsqueeze();
            let attention_mask_bool = attention_mask_tensor.equal_elem(1);
            let logits = model.forward(tokens_tensor, token_type_ids_tensor, attention_mask_bool);
            let logits = logits.clamp(-40.0, 40.0); 
            let logits: Tensor<B, 2> = logits.squeeze_dim(0); // [seq_len, 2]
            let start_logits: Tensor<B, 1> = logits.clone().slice([0..seq_len, 0..1]).squeeze_dim(1);
            let end_logits: Tensor<B, 1> = logits.slice([0..seq_len, 1..2]).squeeze_dim(1);
            let start_probs_tensor = softmax(start_logits, 0);
            let end_probs_tensor = softmax(end_logits, 0);
            let start_probs: Vec<f32> = start_probs_tensor.into_data().convert::<f32>().to_vec()?;
            let end_probs: Vec<f32> = end_probs_tensor.into_data().convert::<f32>().to_vec()?;
            let context_end_offset = context_start_offset + chunk.len();
            for i in context_start_offset..context_end_offset {
                for j in i..(i + 30).min(context_end_offset) {
                    let score = start_probs[i] * end_probs[j];
                    if score > best_score {
                        let answer_tokens = &tokens[i..=j];
                        if let Ok(decoded) = processor.tokenizer.decode(answer_tokens, true) {
                            let clean = decoded.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").trim().to_string();
                            if !clean.is_empty() {
                                best_score = score;
                                best_answer = clean;
                                found_valid_answer = true;
                            }
                        }
                    }
                }
            }
            if window_end == c_ids.len() { break; }
            window_start += stride;
        }
        if found_valid_answer {
            let mut final_answer = best_answer.clone();
            if final_answer.contains("|") {
                let parts: Vec<_> = final_answer.split('|').map(|s| s.trim()).collect();
                let event = parts.iter()
                    .filter(|s| !s.is_empty() && !s.chars().all(|c| c.is_numeric() || c == ')' || c == '('))
                    .last()
                    .map(|s| *s)
                    .unwrap_or_else(|| final_answer.as_str());
                final_answer = event.to_string();
            }
            let norm_score = if best_score.is_finite() {
                best_score.max(0.0).min(1.0)
            } else {
                0.0
            };
            println!("\nQuestion: {}\nAnswer: {}", question, final_answer);
            println!("Confidence Score: {:.2}%", norm_score * 100.0);
        } else {
            println!("\nQuestion: {}\nAnswer: No valid answer found.", question);
        }
    }
    Ok(())
}