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
use std::collections::HashMap;

// Question intent types for better context-aware answering
#[derive(Debug, Clone, Copy)]
pub enum QuestionIntent {
    SingleDateEvent,    // "What on Jan 22?"
    DateRangeEvent,     // "What from Jan 22-25?"
    EventSearch,        // "When is graduation?"
    StatusQuery,        // "What status for July?"
    GenericQuery,       // Fallback
}

/// Classify the intent of a question
pub fn classify_intent(question: &str) -> QuestionIntent {
    let lower = question.to_lowercase();
    
    if lower.contains("status") {
        QuestionIntent::StatusQuery
    } else if lower.contains("when") && (lower.contains("graduation") || lower.contains("ceremony")) {
        QuestionIntent::EventSearch
    } else if lower.contains(" to ") || lower.contains("through") {
        QuestionIntent::DateRangeEvent
    } else if lower.contains("what") || lower.contains("which") || lower.contains("event") {
        QuestionIntent::SingleDateEvent
    } else {
        QuestionIntent::GenericQuery
    }
}

/// Extract date from question text
pub fn extract_date(question: &str) -> Option<String> {
    let months = [
        ("january", "January"), ("february", "February"), ("march", "March"), ("april", "April"),
        ("may", "May"), ("june", "June"), ("july", "July"), ("august", "August"),
        ("september", "September"), ("october", "October"), ("november", "November"), ("december", "December"),
        ("jan", "January"), ("feb", "February"), ("mar", "March"), ("apr", "April"), ("jun", "June"),
        ("jul", "July"), ("aug", "August"), ("sep", "September"), ("oct", "October"), ("nov", "November"), ("dec", "December"),
    ];
    
    let lower = question.to_lowercase();
    
    for (month_str, month_name) in &months {
        if let Some(month_idx) = lower.find(month_str) {
            // Extract day number
            let after_month = &lower[month_idx + month_str.len()..];
            let rest = after_month.trim_start_matches(',').trim_start();
            let day_str: String = rest.chars().take_while(|c| c.is_numeric()).collect();
            
            if !day_str.is_empty() {
                // Extract year if present
                let year_part = rest.chars().skip_while(|c| c.is_numeric()).skip_while(|c| !c.is_numeric()).take_while(|c| c.is_numeric()).collect::<String>();
                
                if !year_part.is_empty() {
                    return Some(format!("{} {}, {}", month_name, day_str, year_part));
                } else {
                    return Some(format!("{} {}", month_name, day_str));
                }
            }
        }
    }
    None
}

/// Semantic date-based answer extraction from context
pub fn find_answer_by_date_matching(question: &str, context: &str) -> Option<String> {
    let date = extract_date(question)?;
    let _intent = classify_intent(question);
    
    // Build a map of "Date X:" patterns to events
    let mut date_events: HashMap<String, Vec<String>> = HashMap::new();
    
    for line in context.lines() {
        if line.contains("Date ") && line.contains(':') {
            // Parse lines like "Date 22: Annual Open Day"
            if let Some(colon_pos) = line.find(':') {
                let before_colon = &line[..colon_pos];
                if let Some(date_start) = before_colon.find("Date ") {
                    let date_part = &before_colon[date_start + 5..].trim();
                    let event = line[colon_pos + 1..].trim().to_string();
                    
                    if !event.is_empty() {
                        date_events.entry(date_part.to_string()).or_insert_with(Vec::new).push(event);
                    }
                }
            }
        }
    }
    
    // Look for exact date match or partial match
    for (date_key, events) in &date_events {
        if date_key.contains(&date) || date.contains(date_key) {
            // Return combined events for this date
            return Some(events.join(" | "));
        }
    }
    
    // Fallback: search directly in context for the date with nearby text
    for line in context.lines() {
        if line.contains(&date) {
            // Extract relevant part of the line
            let clean_line = line.replace("|", " ").trim().to_string();
            if !clean_line.is_empty() {
                return Some(clean_line);
            }
        }
    }
    
    None
}

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

        // --- First, try semantic date-based matching ---
        if let Some(semantic_answer) = find_answer_by_date_matching(question, &context) {
            println!("\nQuestion: {}\nAnswer: {}", question, semantic_answer);
            continue;
        }

        // --- Fallback to neural network inference ---
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
            println!("\nQuestion: {}\nAnswer: {}", question, final_answer);
        } else {
            println!("\nQuestion: {}\nAnswer: No valid answer found.", question);
        }
    }
    Ok(())
}