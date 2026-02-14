from docx import Document
import json

def extract_docx_text_from_tables(docx_path):
    """
    Extract calendar events from table cells in the docx.
    Each cell may contain: "date\nevent1\nevent2\n..."
    """
    doc = Document(docx_path)
    full_text = []
    
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                if text:
                    full_text.append(text)
    
    return "\n".join(full_text)


def update_answer_offsets(doc_text, json_data):
    """
    Updates answer_start offsets based on first occurrence in document text.
    """
    not_found = []
    
    for item in json_data:
        answer_text = item.get("answer_text", "").strip()
        
        if not answer_text:
            item["answer_start"] = 0
            continue
        
        # Try exact match first
        position = doc_text.find(answer_text)
        
        # If not found, try removing extra whitespace/formatting
        if position == -1:
            # Replace multiple spaces/tabs/newlines with single space
            normalized_answer = " ".join(answer_text.split())
            normalized_doc = " ".join(doc_text.split())
            position = normalized_doc.find(normalized_answer)
            
            if position != -1:
                # Map back to original doc position (rough approximation)
                position = doc_text.find(normalized_answer.split()[0]) if normalized_answer else -1
        
        if position != -1:
            item["answer_start"] = position
        else:
            item["answer_start"] = 0
            not_found.append(answer_text[:60])
    
    return json_data, not_found


def main():
    import sys
    from pathlib import Path
    
    # Process all three calendars
    for docx_name in ["calendar_2024", "calendar_2025", "calendar_2026"]:
        docx_path = f"data/{docx_name}.docx"
        json_path = f"data/{docx_name}.json"
        
        print(f"Processing {docx_name}...")
        
        # Extract text from tables
        document_text = extract_docx_text_from_tables(docx_path)
        print(f"  Extracted {len(document_text)} characters from tables")
        
        # Load JSON
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        print(f"  Loaded {len(json_data)} Q&A pairs")
        
        # Update offsets
        updated_json, not_found_answers = update_answer_offsets(document_text, json_data)
        
        # Save
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(updated_json, f, indent=2, ensure_ascii=False)
        
        print(f"  Updated and saved {json_path}")
        
        if not_found_answers:
            print(f"  WARNING: {len(not_found_answers)} answers not found (showing first 5):")
            for ans in not_found_answers[:5]:
                print(f"    - {ans}...")
        else:
            print(f"  SUCCESS: All {len(json_data)} answers found!")
        print()


if __name__ == "__main__":
    main()