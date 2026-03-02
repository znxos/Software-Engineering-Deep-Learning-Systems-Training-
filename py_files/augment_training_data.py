#!/usr/bin/env python3
"""
Generate diverse question phrasings for training data augmentation.
This expands training questions with variations to help the model generalize better.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

def extract_date_from_question(question: str) -> str:
    """Extract date components from question."""
    # Match patterns like "January 22, 2024" or "Jan 22"
    pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}(?:\s*,?\s*\d{4})?)'
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def extract_event_type(question: str) -> str:
    """Extract the type of event being asked about."""
    if 'event' in question.lower():
        return 'event'
    elif 'status' in question.lower():
        return 'status'
    elif 'scheduled' in question.lower():
        return 'scheduled'
    elif 'meeting' in question.lower():
        return 'meeting'
    elif 'committee' in question.lower():
        return 'committee'
    else:
        return 'generic'

def generate_paraphrases(question: str, answer: str) -> List[str]:
    """Generate multiple paraphrases of a question."""
    paraphrases = [question]  # Keep original
    
    date = extract_date_from_question(question)
    if not date:
        return paraphrases
    
    event_type = extract_event_type(question)
    
    # Template variations
    templates = [
        # Original style
        f"What event is scheduled for {date}?",
        f"What is scheduled for {date}?",
        
        # Direct query
        f"{date} - what event?",
        f"{date}: what's on?",
        f"What's on {date}?",
        
        # Information seeking
        f"Tell me what's scheduled for {date}",
        f"Tell me about {date}",
        f"What activities are on {date}?",
        f"What events are on {date}?",
        
        # Alternative phrasings
        f"Which event is on {date}?",
        f"What happens on {date}?",
        f"What's happening on {date}?",
        f"Is there an event on {date}?",
        
        # Status queries
        f"What status is listed for {date}?",
        f"What is the status for {date}?",
        
        # Meeting/Committee specific
        f"What meeting is on {date}?",
        f"Which meetings are scheduled for {date}?",
    ]
    
    # Filter and deduplicate
    for template in templates:
        if template != question and template not in paraphrases:
            paraphrases.append(template)
    
    return paraphrases[:6]  # Limit to 6 variations per question

def augment_qa_data(input_json_path: str, output_json_path: str) -> int:
    """
    Read JSON Q&A data, generate paraphrases, and save augmented data.
    Returns the number of generated questions.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    augmented_data = []
    generated_count = 0
    
    for item in original_data:
        question = item.get('question', '')
        answer_text = item.get('answer_text', '')
        answer_start = item.get('answer_start', 0)
        
        # Keep the original
        augmented_data.append(item)
        
        # Generate paraphrases
        paraphrases = generate_paraphrases(question, answer_text)
        
        for paraphrase in paraphrases[1:]:  # Skip first one (it's the original)
            augmented_item = {
                'question': paraphrase,
                'answer_text': answer_text,
                'answer_start': answer_start
            }
            augmented_data.append(augmented_item)
            generated_count += 1
    
    # Save augmented data
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    return generated_count

if __name__ == "__main__":
    years = [2024, 2025, 2026]
    total_generated = 0
    
    for year in years:
        input_file = f'data/calendar_{year}.json'
        output_file = f'data/calendar_{year}_augmented.json'
        
        if Path(input_file).exists():
            generated = augment_qa_data(input_file, output_file)
            print(f"✓ calendar_{year}: Generated {generated} new question variations")
            print(f"  Original: {input_file}")
            print(f"  Augmented: {output_file}")
            total_generated += generated
        else:
            print(f"⚠ Skipped calendar_{year}.json (not found)")
    
    print(f"\n✓ Total new questions generated: {total_generated}")
    print("\nNext steps:")
    print("1. Review the augmented files: data/calendar_*_augmented.json")
    print("2. Replace original files or merge manually")
    print("3. Retrain the model with augmented data")
