// src/qa_inference.rs
use crate::{
    data::{extract_text_from_docx, QAProcessor},
    training::TrainingConfig,
};
use burn::{
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor},
};
use std::path::Path;

pub fn run_inference<B: Backend>(
    doc_path: String,
    question: String,
    model_path: String,
    device: B::Device,
) {
    // 1. Load configurations from the same file used for training.
    let config: TrainingConfig = serde_json::from_str(
        &std::fs::read_to_string("config.json").expect("Config file 'config.json' not found."),
    )
    .expect("Failed to parse config file.");
    let model_config = config.model;

    // NOTE: Ensure you have the tokenizer file at this path.
    // You can download it from the Hugging Face Hub for 'bert-base-uncased'.
    let tokenizer_path = "data/tokenizer.json";
    let processor = QAProcessor::new(tokenizer_path, model_config.max_seq_length);

    // 2. Load the trained model weights.
    println!("Loading model from {}...", &model_path);
    let model = model_config.init::<B>(&device);
    let model = model
        .load_file(
            &model_path,
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .expect("Failed to load trained model weights. Ensure the model path is correct.");
    println!("Model loaded successfully.");

    // 3. Load and process the document context.
    println!("Processing document: {}", &doc_path);
    let context = extract_text_from_docx(Path::new(&doc_path))
        .expect("Failed to read text from .docx file.");

    // 4. Tokenize the context and question for inference.
    let (encoding, tokens, token_type_ids, attention_mask) = processor
        .process_for_inference(&context, &question)
        .expect("Failed to tokenize input.");

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

    // 7. Find the most likely start and end token indices from the logits.
    let logits: Tensor<B, 2> = logits.squeeze(); // Remove batch dimension.
    let [seq_length, _] = logits.dims();

    let start_logits: Tensor<B, 1> = logits.clone().slice([0..seq_length, 0..1]).squeeze();
    let end_logits: Tensor<B, 1> = logits.slice([0..seq_length, 1..2]).squeeze();

    let start_index = start_logits.argmax(0).into_scalar().elem::<i64>() as usize;
    let end_index = end_logits.argmax(0).into_scalar().elem::<i64>() as usize;

    // 8. Decode the answer tokens back to a string and print.
    let all_token_ids = encoding.get_ids();
    if start_index <= end_index && end_index < all_token_ids.len() {
        let answer_tokens = &all_token_ids[start_index..=end_index];
        let answer = processor.tokenizer.decode(answer_tokens, true).expect("Failed to decode answer tokens.");
        println!("\nQuestion: {}\nAnswer: {}", question, answer);
    } else {
        println!("\nQuestion: {}\nAnswer: Could not find a valid answer in the document.", question);
    }
}