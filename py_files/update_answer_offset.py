# File: data/dot.py
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
import json
import re
from pathlib import Path

def normalize_quotes(text):
    """Normalize smart quotes and common mojibake to straight quotes/chars."""
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    # Mojibake handling (UTF-8 bytes interpreted as Windows-1252/Latin-1)
    text = text.replace('â€™', "'").replace('â€“', "-")
    text = text.replace('â€œ', '"').replace('â€\x9d', '"')
    return text

def extract_docx_text(docx_path):
    """
    Extract text from ALL paragraphs in the docx, including tables and text boxes,
    by iterating over the underlying XML.
    """
    doc = Document(docx_path)
    full_text = []
    
    # Iterate over all <w:p> elements in the document body
    # This catches paragraphs in the body, inside tables, and inside text boxes
    for p_element in doc.element.body.iter(qn('w:p')):
        # Wrap in Paragraph object to access .text property conveniently
        p = Paragraph(p_element, doc)
        text = p.text.strip()
        if text:
            full_text.append(text)
    
    return "\n".join(full_text)

def get_alphanumeric_map(text):
    """
    Returns:
    - clean_text: text with only alphanumeric chars (lowercase)
    - indices: list where indices[i] is the index in original text of the char at clean_text[i]
    """
    clean_chars = []
    indices = []
    for i, char in enumerate(text):
        if char.isalnum():
            clean_chars.append(char.lower())
            indices.append(i)
    return "".join(clean_chars), indices

def update_answer_offsets(doc_text, json_data):
    """
    Updates answer_start offsets based on first occurrence in document text.
    Also updates answer_text to match the document content if fuzzy matched.
    """
    not_found = []
    doc_text_norm = normalize_quotes(doc_text)
    doc_clean, doc_indices = get_alphanumeric_map(doc_text)
    
    for item in json_data:
        answer_text = item.get("answer_text", "").strip()
        
        if not answer_text:
            item["answer_start"] = 0
            continue
        
        idx = -1
        
        # 1. Exact match
        if idx == -1:
            idx = doc_text.find(answer_text)
        
        # 2. Match with normalized quotes
        if idx == -1:
            ans_norm = normalize_quotes(answer_text)
            idx = doc_text_norm.find(ans_norm)
            if idx != -1:
                item["answer_text"] = doc_text[idx:idx+len(answer_text)]
        
        # 3. Flexible match (Regex)
        # Handles:
        # - Comma vs Newline vs Space
        # - Multiple spaces
        # - Smart quotes (already normalized in doc_text_norm)
        if idx == -1:
            ans_norm = normalize_quotes(answer_text)
            # Replace commas with spaces to treat them as separators
            ans_clean = ans_norm.replace(',', ' ')
            tokens = ans_clean.split()
            if tokens:
                escaped_tokens = [re.escape(t) for t in tokens]
                # Allow any whitespace (including newlines) between tokens
                pattern_str = r'\s+'.join(escaped_tokens)
                
                try:
                    match = re.search(pattern_str, doc_text_norm)
                    if match:
                        idx = match.start()
                        item["answer_text"] = doc_text[idx:match.end()]
                except re.error:
                    pass

        # 4. Alphanumeric fuzzy match (Fallback)
        # Handles: "S tart" vs "Start", "(@12:30)" vs "(12:30)", encoding artifacts
        if idx == -1:
            ans_clean, _ = get_alphanumeric_map(answer_text)
            
            if ans_clean and ans_clean in doc_clean:
                clean_start = doc_clean.find(ans_clean)
                clean_end = clean_start + len(ans_clean) - 1
                
                # Map back to original indices
                orig_start = doc_indices[clean_start]
                orig_end = doc_indices[clean_end] + 1 # +1 to include the last char
                
                idx = orig_start
                item["answer_text"] = doc_text[orig_start:orig_end]

        # 5. Multi-part reordering match (Cluster search)
        # Handles: "Event A, Event B" when doc has "Event B, Event A" or "Event A\nEvent B"
        if idx == -1 and ',' in answer_text:
            parts = [p.strip() for p in answer_text.split(',') if p.strip()]
            if len(parts) > 1:
                clean_parts = []
                for p in parts:
                    cp, _ = get_alphanumeric_map(p)
                    if cp: clean_parts.append(cp)
                
                if len(clean_parts) == len(parts):
                    anchor = clean_parts[0]
                    search_start = 0
                    # Heuristic window: 4x length of answer to allow for reordering/newlines
                    window_len = len(answer_text) * 4 
                    
                    while True:
                        anchor_pos = doc_clean.find(anchor, search_start)
                        if anchor_pos == -1:
                            break
                            
                        # Search for other parts near this anchor occurrence
                        w_start = max(0, anchor_pos - window_len)
                        w_end = min(len(doc_clean), anchor_pos + len(anchor) + window_len)
                        
                        current_spans = []
                        # Add anchor span
                        c_end = anchor_pos + len(anchor) - 1
                        current_spans.append((doc_indices[anchor_pos], doc_indices[c_end] + 1))
                        
                        all_found = True
                        for other in clean_parts[1:]:
                            slice_str = doc_clean[w_start:w_end]
                            rel_idx = slice_str.find(other)
                            if rel_idx != -1:
                                abs_idx = w_start + rel_idx
                                o_end = abs_idx + len(other) - 1
                                current_spans.append((doc_indices[abs_idx], doc_indices[o_end] + 1))
                            else:
                                all_found = False
                                break
                        
                        if all_found:
                            min_s = min(s[0] for s in current_spans)
                            max_e = max(s[1] for s in current_spans)
                            idx = min_s
                            item["answer_text"] = doc_text[min_s:max_e]
                            break # Found the best cluster
                        
                        search_start = anchor_pos + 1

        if idx != -1:
            item["answer_start"] = idx
        else:
            item["answer_start"] = 0
            not_found.append(answer_text)
    
    return json_data, not_found


def process_calendar(year):
    """Process calendar data and fix answer offsets."""
    docx_path = Path(f'data/calendar_{year}.docx') # Assuming DOCX files exist
    json_path = Path(f'data/calendar_{year}.json')
    
    print(f"\nProcessing calendar_{year}...")

    # Check if DOCX file exists to extract context
    if not docx_path.exists():
        print(f"  ⚠️ WARNING: DOCX file not found for {year} at {docx_path}. Cannot update offsets without document context.")
        return

    try:
        doc_text = extract_docx_text(docx_path)
    except Exception as e:
        print(f"  ⚠️ ERROR reading docx: {e}")
        return

    if not doc_text:
        print(f"  ⚠️ WARNING: No text extracted from {docx_path}. Cannot update offsets.")
        return

    # Load Q&A data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # The JSON structure is a list of Q&A items, each with 'answer_text'
    if not isinstance(data, list):
        print(f"  ⚠️ WARNING: JSON data for {year} is not a list of Q&A items. Skipping.")
        return
    
    qa_list = data
    
    print(f"  Q&A pairs: {len(qa_list)}")
    
    # Use the existing update_answer_offsets function to fix the offsets
    updated_qa_list, not_found_answers = update_answer_offsets(doc_text, qa_list)
    
    # Save corrected data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_qa_list, f, ensure_ascii=False, indent=2)
    
    updated_count = len(qa_list) - len(not_found_answers)
    print(f"  ✓ Updated {updated_count} answer offsets")
    
    if not_found_answers:
        print(f"  ⚠️ WARNING: {len(not_found_answers)} answers not found (first 5):")
        for ans in not_found_answers[:5]:
            print(f"    - {ans[:60]}...")

if __name__ == "__main__":
    # Process all years
    for year in [2024, 2025, 2026]:
        process_calendar(year)

    print("\n✓ Done!")
