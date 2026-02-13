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

#[derive(Clone)]
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
        let mut tokens_vec = Vec::new();
        let mut token_type_ids_vec = Vec::new();
        let mut attention_mask_vec = Vec::new();
        let mut start_indices_vec = Vec::new();
        let mut end_indices_vec = Vec::new();

        let max_len = items.iter().map(|item| item.tokens.len()).max().unwrap_or(0);
        let batch_size = items.len();

        for item in items {
            let mut item_tokens = item.tokens;
            let mut item_token_type_ids = item.token_type_ids;
            let mut item_attention_mask = item.attention_mask;

            let pad_len = max_len - item_tokens.len();
            item_tokens.extend(vec![0; pad_len]);
            item_token_type_ids.extend(vec![0; pad_len]);
            item_attention_mask.extend(vec![0; pad_len]);

            tokens_vec.extend(item_tokens);
            token_type_ids_vec.extend(item_token_type_ids);
            attention_mask_vec.extend(item_attention_mask);

            let start_idx = item.start_token_idx.min(max_len.saturating_sub(1));
            let end_idx = item.end_token_idx.min(max_len.saturating_sub(1));
            
            start_indices_vec.push(start_idx as i64);
            end_indices_vec.push(end_idx as i64);
        }

        // Create 1D tensors first, then reshape to 2D
        let tokens_1d = Tensor::<BT, 1, Int>::from_data(
            tokens_vec.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(),
            device,
        );
        let tokens_tensor = tokens_1d.reshape([batch_size, max_len]);

        let token_type_ids_1d = Tensor::<BT, 1, Int>::from_data(
            token_type_ids_vec.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(),
            device,
        );
        let token_type_ids_tensor = token_type_ids_1d.reshape([batch_size, max_len]);

        let attention_mask_1d = Tensor::<BT, 1, Int>::from_data(
            attention_mask_vec.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice(),
            device,
        );
        let attention_mask_tensor = attention_mask_1d.reshape([batch_size, max_len])
            .equal_elem(1);

        QABatch {
            tokens: tokens_tensor,
            token_type_ids: token_type_ids_tensor,
            attention_mask: attention_mask_tensor,
            start_indices: Tensor::<BT, 1, Int>::from_data(start_indices_vec.as_slice(), device),
            end_indices: Tensor::<BT, 1, Int>::from_data(end_indices_vec.as_slice(), device),
        }
    }
}
