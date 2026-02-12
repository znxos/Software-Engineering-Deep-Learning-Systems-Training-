// src/data.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};
use docx_rs::read_docx;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};

// Represents a single Q&A item
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QAItem {
    pub context: String,
    pub question: String,
    pub answer_start: usize, // Character index of answer start in context
    pub answer_text: String,
}

// Represents a tokenized item ready for the model
#[derive(Clone, Debug)]
pub struct QABatchItem {
    pub tokens: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub start_token_idx: usize,
    pub end_token_idx: usize,
}

/// Extracts text from a .docx file.
pub fn extract_text_from_docx(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    let docx = read_docx(&buf).map_err(|e| anyhow::anyhow!("Failed to parse .docx: {}", e))?;

    let mut text_content = String::new();
    for child in docx.document.children {
        if let docx_rs::DocumentChild::Paragraph(p) = child {
            let mut paragraph_text = String::new();
            for p_child in p.children {
                if let docx_rs::ParagraphChild::Run(run) = p_child {
                    for r_child in run.children {
                        if let docx_rs::RunChild::Text(text) = r_child {
                            paragraph_text.push_str(&text.text);
                        }
                    }
                }
            }
            text_content.push_str(&paragraph_text);
            text_content.push('\n'); // Add newlines between paragraphs
        }
    }
    Ok(text_content)
}

/// Processes raw data into tokenized items for the Q&A model.
pub struct QAProcessor {
    pub tokenizer: Tokenizer,
    _max_length: usize,
}

impl QAProcessor {
    pub fn new(tokenizer_path: &str, max_length: usize) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        Self { tokenizer, _max_length: max_length }
    }

    /// Tokenizes a context and question for inference.
    pub fn process_for_inference(
        &self,
        context: &str,
        question: &str,
    ) -> Option<(Encoding, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let encoding = self.tokenizer.encode((question.to_string(), context.to_string()), false).ok()?;
        let tokens = encoding.get_ids().iter().map(|&x| x as u32).collect();
        let token_type_ids = encoding.get_type_ids().iter().map(|&x| x as u32).collect();
        let attention_mask = encoding.get_attention_mask().iter().map(|&x| x as u32).collect();
        Some((encoding, tokens, token_type_ids, attention_mask))
    }

    pub fn process(&self, item: &QAItem) -> Option<QABatchItem> {
        let final_encoding = self.tokenizer
            .encode((item.question.clone(), item.context.clone()), true)
            .ok()?;

        let answer_char_end = item.answer_start + item.answer_text.len();

        // The context is the second part of the pair, so its sequence ID is 1.
        // The char_to_token method will correctly find the token index within the combined sequence.
        let start_token_idx = final_encoding.char_to_token(item.answer_start, 1)?;
        let end_token_idx = final_encoding.char_to_token(answer_char_end.saturating_sub(1), 1)?;

        Some(QABatchItem {
            tokens: final_encoding.get_ids().iter().map(|&x| x as u32).collect(),
            token_type_ids: final_encoding.get_type_ids().iter().map(|&x| x as u32).collect(),
            attention_mask: final_encoding.get_attention_mask().iter().map(|&x| x as u32).collect(),
            start_token_idx,
            end_token_idx,
        })
    }
}

/// Loads a dataset by finding all .docx files in a directory and pairing them
/// with a corresponding .json file that contains the questions and answers.
pub fn load_dataset_from_data_folder(data_path: &str) -> anyhow::Result<Vec<QAItem>> {
    let mut all_items = Vec::new();
    let entries = fs::read_dir(data_path)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("docx") {
            let docx_path = path;
            let json_path = docx_path.with_extension("json");

            if json_path.exists() {
                println!("Loading training data from: {:?}", &docx_path);
                let context = extract_text_from_docx(&docx_path)?;
                let json_content = fs::read_to_string(&json_path)?;

                #[derive(Deserialize)]
                struct RawQAItem {
                    question: String,
                    answer_start: usize,
                    answer_text: String,
                }

                let raw_items: Vec<RawQAItem> = serde_json::from_str(&json_content)?;

                for raw_item in raw_items {
                    all_items.push(QAItem {
                        context: context.clone(),
                        question: raw_item.question,
                        answer_start: raw_item.answer_start,
                        answer_text: raw_item.answer_text,
                    });
                }
            }
        }
    }
    Ok(all_items)
}

#[derive(Clone, Debug)]
pub struct QABatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub token_type_ids: Tensor<B, 2, Int>,
    pub attention_mask: Tensor<B, 2, Bool>,
    pub start_indices: Tensor<B, 1, Int>,
    pub end_indices: Tensor<B, 1, Int>,
}

pub struct QABatcher<B: Backend> {
    _device: B::Device,
}

impl<B: Backend> QABatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { _device: device }
    }
}

impl<BT: Backend> Batcher<BT, QABatchItem, QABatch<BT>> for QABatcher<BT> {
    fn batch(&self, items: Vec<QABatchItem>, device: &BT::Device) -> QABatch<BT> {
        let mut tokens_vec: Vec<Tensor<BT, 1, Int>> = Vec::new();
        let mut token_type_ids_vec: Vec<Tensor<BT, 1, Int>> = Vec::new();
        let mut attention_masks_vec: Vec<Tensor<BT, 1, Int>> = Vec::new();
        let mut start_indices_vec: Vec<i64> = Vec::new();
        let mut end_indices_vec: Vec<i64> = Vec::new();

        // Find max length in the batch for padding
        let max_len = items.iter().map(|item| item.tokens.len()).max().unwrap_or(0);

        for mut item in items {
            start_indices_vec.push(item.start_token_idx as i64);
            end_indices_vec.push(item.end_token_idx as i64);

            // Pad with 0s (assuming 0 is the padding token ID)
            let pad_size = max_len - item.tokens.len();
            item.tokens.extend(vec![0; pad_size]);
            item.token_type_ids.extend(vec![0; pad_size]);
            item.attention_mask.extend(vec![0; pad_size]);

            let tokens_data = item.tokens.into_iter().map(|t| t as i64).collect::<Vec<i64>>();
            tokens_vec.push(Tensor::<BT, 1, Int>::from_data(tokens_data.as_slice(), device));

            let token_type_ids_data = item.token_type_ids.into_iter().map(|t| t as i64).collect::<Vec<i64>>();
            token_type_ids_vec.push(Tensor::<BT, 1, Int>::from_data(token_type_ids_data.as_slice(), device));

            let attention_mask_data = item.attention_mask.into_iter().map(|t| t as i64).collect::<Vec<i64>>();
            attention_masks_vec.push(Tensor::<BT, 1, Int>::from_data(attention_mask_data.as_slice(), device));
        }

        // Stack the tensors to create a batch and move to the correct device
        let tokens = Tensor::stack(tokens_vec, 0).to_device(device);
        let token_type_ids = Tensor::stack(token_type_ids_vec, 0).to_device(device);
        let attention_mask_int = Tensor::stack(attention_masks_vec, 0).to_device(device);

        // Convert attention mask to boolean (where 1 is true, 0 is false)
        let attention_mask = attention_mask_int.equal_elem(1);

        let start_indices = Tensor::<BT, 1, Int>::from_data(start_indices_vec.as_slice(), device);
        let end_indices = Tensor::<BT, 1, Int>::from_data(end_indices_vec.as_slice(), device);

        QABatch {
            tokens,
            token_type_ids,
            attention_mask,
            start_indices,
            end_indices,
        }
    }
}
