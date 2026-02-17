// src/data.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};
use docx_rs::{read_docx, DocumentChild, ParagraphChild, RunChild, TableCellContent, TableChild, TableRowChild};
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

pub fn extract_text_from_docx(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    let docx = read_docx(&buf).map_err(|e| anyhow::anyhow!("Failed to parse .docx: {}", e))?;

    let mut full_text = Vec::new();

    fn extract_paragraph_text(p: &docx_rs::Paragraph) -> String {
        let mut text = String::new();
        for p_child in &p.children {
            if let ParagraphChild::Run(run) = p_child {
                for r_child in &run.children {
                    if let RunChild::Text(t) = r_child {
                        text.push_str(&t.text);
                    }
                }
            }
        }
        text
    }

    // Extract text from all paragraphs in the document
    for child in docx.document.children {
        match child {
            DocumentChild::Paragraph(p) => {
                let text = extract_paragraph_text(&p).trim().to_string();
                if !text.is_empty() {
                    full_text.push(text);
                }
            }
            DocumentChild::Table(table) => {
                for row in table.rows {
                    let TableChild::TableRow(tr) = row;
                    for cell in tr.cells {
                        let TableRowChild::TableCell(tc) = cell;
                        for content in tc.children {
                            if let TableCellContent::Paragraph(p) = content {
                                let text = extract_paragraph_text(&p).trim().to_string();
                                if !text.is_empty() {
                                    full_text.push(text);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    Ok(full_text.join("\n"))
}

/// Processes raw data into tokenized items for the Q&A model.
pub struct QAProcessor {
    pub tokenizer: Tokenizer,
    max_length: usize,
}

impl QAProcessor {
    pub fn new(tokenizer_path: &str, max_length: usize) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        Self { tokenizer, max_length }
    }

    /// Tokenizes a context and question for inference.
    pub fn process_for_inference(
        &self,
        context: &str,
        question: &str,
    ) -> Option<(Encoding, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let encoding = self.tokenizer.encode((question.to_string(), context.to_string()), false).ok()?;
        let mut tokens: Vec<u32> = encoding.get_ids().iter().map(|&x| x as u32).collect();
        let mut token_type_ids: Vec<u32> = encoding.get_type_ids().iter().map(|&x| x as u32).collect();
        let mut attention_mask: Vec<u32> = encoding.get_attention_mask().iter().map(|&x| x as u32).collect();

        // Truncate to max_length to avoid model panic
        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length);
            token_type_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
        }

        Some((encoding, tokens, token_type_ids, attention_mask))
    }

    pub fn process(&self, item: &QAItem) -> Option<QABatchItem> {
        // 1. Tokenize Question and Context separately to handle windowing manually
        let q_encoding = self.tokenizer.encode(item.question.clone(), false).ok()?;
        let c_encoding = self.tokenizer.encode(item.context.clone(), false).ok()?;

        let q_ids = q_encoding.get_ids();
        let c_ids = c_encoding.get_ids();

        let answer_char_end = item.answer_start + item.answer_text.len();
        
        // Map character positions to token positions in the context
        let start_token_idx_c = c_encoding.char_to_token(item.answer_start, 0)?;
        let end_token_idx_c = c_encoding.char_to_token(answer_char_end.saturating_sub(1), 0)?;

        // 2. Create a window around the answer
        // Format: [CLS] Question [SEP] Context_Window [SEP]
        // Max context length = max_seq_len - q_len - 3 (special tokens)
        let max_context_len = self.max_length.saturating_sub(q_ids.len() + 3);
        
        // Determine window bounds centered around the answer if possible
        let _answer_len = end_token_idx_c - start_token_idx_c + 1;
        let half_window = max_context_len / 2;
        
        let mut window_start = start_token_idx_c.saturating_sub(half_window);
        let mut window_end = window_start + max_context_len;

        if window_end > c_ids.len() {
            window_end = c_ids.len();
            window_start = window_end.saturating_sub(max_context_len);
        }

        let context_window = &c_ids[window_start..window_end];

        // 3. Construct final token sequence
        let mut tokens = Vec::new();
        let mut token_type_ids = Vec::new();

        // [CLS] Question [SEP]
        tokens.push(101); // CLS
        tokens.extend_from_slice(q_ids);
        tokens.push(102); // SEP
        let context_start_offset = tokens.len();

        // Context Window [SEP]
        tokens.extend_from_slice(context_window);
        tokens.push(102); // SEP

        // Token Type IDs: 0 for Question, 1 for Context
        token_type_ids.extend(vec![0; context_start_offset]);
        token_type_ids.extend(vec![1; tokens.len() - context_start_offset]);

        // Calculate new answer positions relative to the final sequence
        let new_start_idx = context_start_offset + (start_token_idx_c - window_start);
        let new_end_idx = context_start_offset + (end_token_idx_c - window_start);

        // Attention mask is all 1s for valid tokens (padding handled by batcher)
        let attention_mask = vec![1; tokens.len()];

        Some(QABatchItem {
            tokens: tokens.iter().map(|&x| x as u32).collect(),
            token_type_ids: token_type_ids.iter().map(|&x| x as u32).collect(),
            attention_mask: attention_mask.iter().map(|&x| x as u32).collect(),
            start_token_idx: new_start_idx,
            end_token_idx: new_end_idx,
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
                    // Re-find the answer in the Rust-extracted text to ensure indices are correct
                    // The Python extraction might differ slightly from Rust extraction
                    let actual_start = if let Some(idx) = context.find(&raw_item.answer_text) {
                        idx
                    } else {
                        // If exact match fails, fallback to the provided index (might be risky)
                        raw_item.answer_start
                    };

                    all_items.push(QAItem {
                        context: context.clone(),
                        question: raw_item.question,
                        answer_start: actual_start,
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
