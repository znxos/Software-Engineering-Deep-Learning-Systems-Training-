// src/data.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};
use docx_rs::{read_docx, DocumentChild, ParagraphChild, RunChild, TableCellContent, TableChild, TableRowChild, InsertChild};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use tokenizers::Tokenizer;

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

    // Extract text from all document children
    for child in docx.document.children {
        match child {
            DocumentChild::Paragraph(p) => {
                let text = extract_paragraph_text(&p).trim().to_string();
                if !text.is_empty() {
                    full_text.push(text);
                }
            }
            DocumentChild::Table(table) => {
                // Extract all table content
                let table_text = extract_table_text(&table);
                for row in table_text {
                    if !row.is_empty() {
                        full_text.push(row);
                    }
                }
            }
            DocumentChild::StructuredDataTag(sdt) => {
                let sdt_text = extract_sdt_text(&sdt);
                for item in sdt_text {
                    if !item.is_empty() {
                        full_text.push(item);
                    }
                }
            }
            DocumentChild::Section(_) => {
                // Sections are handled via their children in other elements
            }
            _ => {}
        }
    }
    
    Ok(full_text.join("\n"))
}

fn extract_paragraph_text(p: &docx_rs::Paragraph) -> String {
    let mut text = String::new();
    for p_child in &p.children {
        text.push_str(&extract_paragraph_child_text(p_child));
    }
    text
}

fn extract_paragraph_child_text(child: &ParagraphChild) -> String {
    match child {
        ParagraphChild::Run(run) => extract_run_text(run),
        ParagraphChild::Hyperlink(link) => {
            let mut text = String::new();
            for child in &link.children {
                text.push_str(&extract_paragraph_child_text(child));
            }
            text
        },
        ParagraphChild::Insert(insert) => {
            let mut text = String::new();
            for child in &insert.children {
                if let InsertChild::Run(run) = child {
                    text.push_str(&extract_run_text(run));
                }
            }
            text
        },
        ParagraphChild::StructuredDataTag(sdt) => {
            extract_sdt_text(sdt).join(" ")
        },
        _ => String::new(),
    }
}

fn extract_run_text(run: &docx_rs::Run) -> String {
    let mut text = String::new();
    for r_child in &run.children {
        match r_child {
            RunChild::Text(t) => text.push_str(&t.text),
            RunChild::Tab(_) => text.push('\t'),
            RunChild::Break(_) => text.push('\n'),
            _ => {}
        }
    }
    text
}


fn extract_table_text(table: &docx_rs::Table) -> Vec<String> {
    // Check if this is a calendar-like table with date headers
    if is_calendar_table(table) {
        return extract_calendar_table(table);
    }

    // Standard table extraction for non-calendar tables
    let mut table_rows_text = Vec::new();
    for row in &table.rows {
        let TableChild::TableRow(tr) = row;
        let mut row_cells = Vec::new();
        for cell in &tr.cells {
            let TableRowChild::TableCell(tc) = cell;
            let mut cell_content = Vec::new();
            let mut nested_tables = Vec::new();
            
            // First pass: collect all content including nested tables
            for content in &tc.children {
                match content {
                    TableCellContent::Paragraph(p) => {
                        let text = extract_paragraph_text(p).trim().to_string();
                        if !text.is_empty() {
                            cell_content.push(text);
                        }
                    }
                    TableCellContent::Table(nested_table) => {
                        // Extract nested table with preserved structure
                        let nested_text = extract_table_text(nested_table);
                        if !nested_text.is_empty() {
                            nested_tables.push(nested_text.join("\n"));
                        }
                    }
                    TableCellContent::StructuredDataTag(sdt) => {
                        let sdt_text = extract_sdt_text(&sdt);
                        if !sdt_text.is_empty() {
                            cell_content.push(sdt_text.join("\n"));
                        }
                    }
                    _ => {}
                }
            }
            
            // Combine cell content with nested tables
            let mut full_cell_content = cell_content;
            full_cell_content.extend(nested_tables);
            
            if !full_cell_content.is_empty() {
                row_cells.push(full_cell_content.join(" | "));
            }
        }
        if !row_cells.is_empty() {
            table_rows_text.push(row_cells.join(" | "));
        }
    }
    table_rows_text
}

// Check if table is a calendar-like structure (has numeric dates as headers)
fn is_calendar_table(table: &docx_rs::Table) -> bool {
    if table.rows.is_empty() {
        return false;
    }

    // Check first row for numeric date values
    let first_row = &table.rows[0];
    let TableChild::TableRow(tr) = first_row;
    
    let mut numeric_count = 0;
    let mut total_cells = 0;

    for cell in &tr.cells {
        let TableRowChild::TableCell(tc) = cell;
        let cell_text = extract_cell_text(tc).trim().to_string();
        total_cells += 1;
            
        // Check if cell contains only numbers (dates like 1, 2, 3... or 22, 23, 24)
        if !cell_text.is_empty() && cell_text.chars().all(|c| c.is_numeric() || c.is_whitespace()) {
            numeric_count += 1;
        }
    }

    // If at least 50% of cells are numeric, likely a calendar
    total_cells > 0 && numeric_count as f32 / total_cells as f32 > 0.5
}

// Extract calendar table with date-to-event mapping
fn extract_calendar_table(table: &docx_rs::Table) -> Vec<String> {
    let mut result = Vec::new();
    
    if table.rows.is_empty() {
        return result;
    }

    // Extract date headers from first row
    let mut dates: Vec<String> = Vec::new();
    let first_row = &table.rows[0];
    let TableChild::TableRow(tr) = first_row;
    
    for cell in &tr.cells {
        let TableRowChild::TableCell(tc) = cell;
        let cell_text = extract_cell_text(tc).trim().to_string();
        dates.push(cell_text);
    }

    // Process subsequent rows (events)
    for (_row_idx, row) in table.rows.iter().enumerate().skip(1) {
        let TableChild::TableRow(tr) = row;
        let mut col_idx = 0;
        let mut current_span = 1; // Track column span for merged cells

        for cell in &tr.cells {
            let TableRowChild::TableCell(tc) = cell;
            let cell_content = extract_cell_text_detailed(tc);
            
            // Check if this is a merged cell (empty and previous cell had content)
            let is_empty = cell_content.trim().is_empty();
            
            if !is_empty {
                // Determine column span (heuristic: check cell width)
                current_span = detect_column_span(tc, &dates.len());
                
                // Map this event to all columns it spans
                for span_offset in 0..current_span {
                    let target_col = col_idx + span_offset;
                    if target_col < dates.len() {
                        let date = &dates[target_col];
                        // Create explicit date-to-event mapping
                        result.push(format!("Date {}: {}", date, cell_content));
                    }
                }
            } else if current_span > 1 {
                // This is a merged cell continuation - event was already recorded above
                col_idx += current_span;
                continue;
            }
            
            col_idx += 1;
        }
    }

    // If no explicit mappings were found, fall back to standard extraction
    if result.is_empty() {
        let mut standard_rows = Vec::new();
        for row in &table.rows {
            let TableChild::TableRow(tr) = row;
            let mut row_cells = Vec::new();
            for cell in &tr.cells {
                let TableRowChild::TableCell(tc) = cell;
                let cell_text = extract_cell_text(tc);
                if !cell_text.trim().is_empty() {
                    row_cells.push(cell_text);
                }
            }
            if !row_cells.is_empty() {
                standard_rows.push(row_cells.join(" | "));
            }
        }
        return standard_rows;
    }

    result
}

// Extract cell text without detailed formatting
fn extract_cell_text(cell: &docx_rs::TableCell) -> String {
    let mut text = String::new();
    for content in &cell.children {
        match content {
            TableCellContent::Paragraph(p) => {
                text.push_str(&extract_paragraph_text(p));
                text.push(' ');
            }
            TableCellContent::Table(nested_table) => {
                let nested = extract_table_text(nested_table);
                text.push_str(&nested.join(" "));
                text.push(' ');
            }
            _ => {}
        }
    }
    text.trim().to_string()
}

// Extract cell text with all details (paragraphs, nested tables)
fn extract_cell_text_detailed(cell: &docx_rs::TableCell) -> String {
    let mut paragraphs = Vec::new();
    let mut nested_tables = Vec::new();

    for content in &cell.children {
        match content {
            TableCellContent::Paragraph(p) => {
                let text = extract_paragraph_text(p).trim().to_string();
                if !text.is_empty() {
                    paragraphs.push(text);
                }
            }
            TableCellContent::Table(nested_table) => {
                let nested = extract_table_text(nested_table);
                if !nested.is_empty() {
                    nested_tables.push(nested.join(" | "));
                }
            }
            _ => {}
        }
    }

    let mut result = paragraphs.join("\n");
    if !nested_tables.is_empty() {
        if !result.is_empty() {
            result.push_str("\n");
        }
        result.push_str(&nested_tables.join("\n"));
    }
    result
}

// Detect how many columns a cell spans (heuristic based on cell width)
fn detect_column_span(cell: &docx_rs::TableCell, _total_cols: &usize) -> usize {
    // Check tcW (table cell width) - rough heuristic
    // If cell width is roughly 2x or 3x average, it spans multiple columns
    // For now, use simple heuristic: 1 column by default, 2+ if explicitly marked
    
    // Default to 1 unless we can detect it spans multiple
    for content in &cell.children {
        if let TableCellContent::Paragraph(p) = content {
            let text = extract_paragraph_text(p);
            // If paragraph contains multiple date-like patterns, might span
            if text.matches(|c: char| c.is_numeric()).count() > 2 {
                return 2; // Conservative estimate
            }
        }
    }
    1
}

fn extract_sdt_text(sdt: &docx_rs::StructuredDataTag) -> Vec<String> {
    let mut sdt_text = Vec::new();
    for child in &sdt.children {
        match child {
            docx_rs::StructuredDataTagChild::Paragraph(p) => {
                let text = extract_paragraph_text(p).trim().to_string();
                if !text.is_empty() {
                    sdt_text.push(text);
                }
            }
            docx_rs::StructuredDataTagChild::Table(t) => {
                let table_text = extract_table_text(t);
                if !table_text.is_empty() {
                    sdt_text.push(table_text.join("\n"));
                }
            }
            docx_rs::StructuredDataTagChild::StructuredDataTag(nested_sdt) => {
                let nested_text = extract_sdt_text(nested_sdt);
                if !nested_text.is_empty() {
                    sdt_text.push(nested_text.join("\n"));
                }
            }
            docx_rs::StructuredDataTagChild::Run(run) => {
                let text = extract_run_text(run);
                if !text.is_empty() {
                    sdt_text.push(text);
                }
            }
            _ => {}
        }
    }
    sdt_text
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

// Helper to find answer in text with multiple matching strategies
fn find_answer_in_context(context: &str, answer: &str) -> Option<(usize, String)> {
    // 0. Check for range question patterns (e.g., "April 22 to April 23")
    if let Some(result) = find_range_answer(context, answer) {
        return Some(result);
    }

    // 1. Exact match
    if let Some(idx) = context.find(answer) {
        return Some((idx, answer.to_string()));
    }

    // 2. Whitespace insensitive match (handles newlines, tabs, multiple spaces)
    if let Some(result) = find_with_flexible_whitespace(context, answer) {
        return Some(result);
    }

    // 3. Pipe-separated table match (handles table cells)
    if answer.contains('|') || context.contains('|') {
        if let Some(result) = find_in_table_format(context, answer) {
            return Some(result);
        }
    }

    // 4. Case-insensitive match
    let context_lower = context.to_lowercase();
    let answer_lower = answer.to_lowercase();
    if let Some(idx) = context_lower.find(&answer_lower) {
        return Some((idx, context[idx..idx + answer.len()].to_string()));
    }

    // 5. Alphanumeric match (ignores all whitespace/symbols)
    if let Some(result) = find_alphanumeric_match(context, answer) {
        return Some(result);
    }

    // 6. Partial match strategies
    if let Some(result) = find_partial_match(context, answer) {
        return Some(result);
    }

    // 7. Parentheses stripping fallback
    if let Some(paren_idx) = answer.find('(') {
        let truncated = answer[..paren_idx].trim();
        if truncated.len() > 3 {
            return find_answer_in_context(context, truncated);
        }
    }

    None
}

// Find answer in range format (e.g., "April 22 to April 23" -> find spanning event)
fn find_range_answer(context: &str, question: &str) -> Option<(usize, String)> {
    // Extract date range from question (e.g., "April 22 to April 23" or "June 22 to June 25")
    let dates = extract_date_range(question)?;
    if dates.is_empty() {
        return None;
    }

    let (start_date, end_date) = (&dates[0], &dates[dates.len() - 1]);

    // Look for events that span from start_date to across end_date
    // Pattern: "Date <start>: <event> ... Date <end>: <event>" or a single spanning event
    
    // Use simple string search instead of regex for efficiency
    for line in context.lines() {
        if line.contains(&format!("Date {}", start_date)) {
            // This line has the start date, extract event
            if let Some(colon_idx) = line.find(':') {
                let event_start = colon_idx + 1;
                let event = line[event_start..].trim();
                
                if !event.is_empty() {
                    // Check if same event appears for end date too
                    for end_line in context.lines() {
                        if end_line.contains(&format!("Date {}", end_date)) {
                            if let Some(end_colon) = end_line.find(':') {
                                let end_event = end_line[end_colon + 1..].trim();
                                
                                // If events match or pattern suggests spanning
                                if end_event == event || end_event.is_empty() {
                                    let search_start = format!("Date {}", start_date);
                                    if let Some(idx) = context.find(&search_start) {
                                        return Some((idx, event.to_string()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// Extract date range from question (e.g., "April 22 to April 23" -> ["April 22", "April 23"])
fn extract_date_range(question: &str) -> Option<Vec<String>> {
    let lower = question.to_lowercase();
    
    // Look for "to" pattern
    if !lower.contains(" to ") {
        return None;
    }

    let parts: Vec<&str> = lower.split(" to ").collect();
    if parts.len() != 2 {
        return None;
    }

    let mut dates = Vec::new();

    // Extract date from before "to"
    if let Some(date1) = extract_date_from_text(parts[0]) {
        dates.push(date1);
    }

    // Extract date from after "to"  
    if let Some(date2) = extract_date_from_text(parts[1]) {
        dates.push(date2);
    }

    if dates.len() == 2 {
        Some(dates)
    } else {
        None
    }
}

// Extract date pattern from text (e.g., "April 22" or "March 19")
fn extract_date_from_text(text: &str) -> Option<String> {
    let months = ["january", "february", "march", "april", "may", "june", 
                  "july", "august", "september", "october", "november", "december",
                  "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    
    for month in &months {
        if text.contains(month) {
            // Find the month word position
            if let Some(month_idx) = text.find(month) {
                // Extract month and following day number
                let after_month = &text[month_idx..];
                let month_word = if *month == "january" { "January" }
                    else if *month == "february" { "February" }
                    else if *month == "march" { "March" }
                    else if *month == "april" { "April" }
                    else if *month == "may" { "May" }
                    else if *month == "june" { "June" }
                    else if *month == "july" { "July" }
                    else if *month == "august" { "August" }
                    else if *month == "september" { "September" }
                    else if *month == "october" { "October" }
                    else if *month == "november" { "November" }
                    else { "December" };

                // Extract day number following month
                let rest = &after_month[month.len()..];
                let day_part = rest.trim_start();
                let day_str: String = day_part.chars()
                    .take_while(|c| c.is_numeric())
                    .collect();
                
                if !day_str.is_empty() {
                    return Some(format!("{} {}", month_word, day_str));
                }
            }
        }
    }
    
    None
}

// Try to match with flexible whitespace (newlines, tabs, multiple spaces)
fn find_with_flexible_whitespace(context: &str, answer: &str) -> Option<(usize, String)> {
    let answer_parts: Vec<&str> = answer.split_whitespace().collect();
    if answer_parts.is_empty() {
        return None;
    }

    let first_part = answer_parts[0];
    let mut search_start = 0;

    while let Some(start_idx) = context[search_start..].find(first_part) {
        let abs_start = search_start + start_idx;
        let mut current_idx = abs_start + first_part.len();
        let mut match_failed = false;
        let matched_text_start = abs_start;

        for part in &answer_parts[1..] {
            if current_idx >= context.len() {
                match_failed = true;
                break;
            }

            let remainder = &context[current_idx..];
            let trimmed = remainder.trim_start();
            let skipped = remainder.len() - trimmed.len();
            current_idx += skipped;

            if current_idx >= context.len() || !context[current_idx..].starts_with(part) {
                match_failed = true;
                break;
            }
            current_idx += part.len();
        }

        if !match_failed {
            return Some((matched_text_start, context[matched_text_start..current_idx].to_string()));
        }
        search_start = abs_start + 1;
    }

    None
}

// Match in table format (cells separated by |)
fn find_in_table_format(context: &str, answer: &str) -> Option<(usize, String)> {
    // Split by pipe to look for table cells
    let cells: Vec<&str> = context.split('|').collect();
    
    for (cell_idx, cell) in cells.iter().enumerate() {
        if let Some(idx) = cell.find(answer) {
            // Calculate absolute position in original context
            let mut abs_pos = 0;
            for (i, c) in cells.iter().enumerate() {
                if i < cell_idx {
                    abs_pos += c.len() + 1; // +1 for the pipe
                }
            }
            abs_pos += idx;
            return Some((abs_pos, answer.to_string()));
        }
    }

    None
}

// Alphanumeric fuzzy match
fn find_alphanumeric_match(context: &str, answer: &str) -> Option<(usize, String)> {
    let answer_clean: String = answer
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();

    if answer_clean.is_empty() {
        return None;
    }

    let mut context_clean = String::with_capacity(context.len());
    let mut indices_map = Vec::with_capacity(context.len());

    for (idx, c) in context.char_indices() {
        if c.is_alphanumeric() {
            context_clean.push(c.to_lowercase().next().unwrap());
            indices_map.push(idx);
        }
    }

    if let Some(start_clean_idx) = context_clean.find(&answer_clean) {
        let end_clean_idx = start_clean_idx + answer_clean.len() - 1;

        if start_clean_idx < indices_map.len() && end_clean_idx < indices_map.len() {
            let start_original_idx = indices_map[start_clean_idx];
            let end_original_start = indices_map[end_clean_idx];
            let end_char_len = context[end_original_start..]
                .chars()
                .next()
                .unwrap()
                .len_utf8();
            let end_original_idx = end_original_start + end_char_len;

            return Some((start_original_idx, context[start_original_idx..end_original_idx].to_string()));
        }
    }

    None
}

// Partial match: find answer by searching for key words
fn find_partial_match(context: &str, answer: &str) -> Option<(usize, String)> {
    let words: Vec<&str> = answer
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    if words.is_empty() {
        return None;
    }

    // Use the longest word as anchor for better matching
    let anchor = words.iter().max_by_key(|w| w.len()).unwrap();

    if let Some(_idx) = context.find(anchor) {
        // Expand around the anchor to find surrounding context
        let context_lower = context.to_lowercase();
        let anchor_lower = anchor.to_lowercase();

        if let Some(lower_idx) = context_lower.find(&anchor_lower) {
            return Some((lower_idx, context[lower_idx..].lines().next().unwrap_or("").to_string()));
        }
    }

    None
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
                    answer_text: String,
                }

                let raw_items: Vec<RawQAItem> = serde_json::from_str(&json_content)?;

                let mut last_pos = 0;
                for raw_item in raw_items {
                    if raw_item.answer_text.trim().is_empty() { continue; }

                    // Try finding chronologically first (search after the last found item)
                    let search_slice = &context[last_pos..];
                    let found = find_answer_in_context(search_slice, &raw_item.answer_text)
                        .map(|(idx, text)| (last_pos + idx, text));

                    let (actual_start, actual_text) = if let Some(res) = found {
                        res
                    } else {
                        // Fallback: Search from beginning if not found chronologically
                        if let Some(res) = find_answer_in_context(&context, &raw_item.answer_text) {
                            res
                        } else {
                            println!("Warning: Could not find answer for question '{}'", raw_item.question);
                            continue;
                        }
                    };

                    if actual_start >= last_pos { last_pos = actual_start + actual_text.len(); }

                    all_items.push(QAItem {
                        context: context.clone(),
                        question: raw_item.question,
                        answer_start: actual_start,
                        answer_text: actual_text,
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
