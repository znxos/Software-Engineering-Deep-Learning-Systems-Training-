
5

Automatic Zoom
SEG 580S: Software Engineering Deep
Learning Systems Training on CPUT Data
Assignment
Preamble:
This assignment is designed to test your ability to build a complete machine learning system
from scratch using Rust and You ARE ALLOWED TO USE ALL THE AI ASSISTANCE YOU CAN
GET IN THE WORLD TO MAKE THIS WORK, OPEN SOURCE OR PAID. The goal is to
demonstrate your understanding of the entire ML pipeline, from data processing to model
training and deployment. You will be evaluated on both the functionality of your code and the
clarity of your project report.
Question and Answering System with Rust and
Burn Framework
Due Date: 2 weeks from assignment date
Marks: 200 (to be scaled to 100% on Blackboard)
Submission: GitHub repository link + Project report in Markdown
Assignment Overview
Build a complete Question-Answering (Q&A) system that reads Word documents attached
and answers questions about their content using Rust and the Burn deep learning
framework. You will implement the full ML pipeline from data loading to model deployment.
Question to be answered are like "What is the Month and date will the 2026 End of year
Graduation Ceremony be held?" or "How many times did the HDC hold their meetings in 2024".
What You'll Build
A working system that:
1. Loads and processes Word documents ( .docx  files)
2. Trains a transformer-based neural network
3. Answers natural language questions about the documents
4. Can be run via command-line interface
Learning Objectives
Implement a complete ML pipeline end-to-end
A (180-200): Exceptional - All requirements exceeded, polished delivery
B (160-179): Good - All core requirements met, minor issues
C (140-159): Acceptable - Core functionality works, some gaps
D (120-139): Minimal - Basic functionality, significant issues
F (<120): Incomplete - Missing major components or non-functional
Resources and Starter Materials
Required Dependencies
Add to your  Cargo.toml :
PLEASE PLEASE: Do Not Change the versions of the dependencies, as we will be using the same
versions for grading. You can add additional dependencies if needed, but these are the core ones
you must use. IF YOU DO CHANGE, YOU WILL GET A ZERO MARK FOR EVERYTHING.
[package]   name = "word-doc-qa"   version = "0.1.0"   edition = "2021"   [dependencies]   burn = { version = "0.20.1", features = ["train", "wgpu", "autodiff"] }   docx-rs = "0.4"   tokenizers = "0.15"   serde = { version = "1.0", features = ["derive"] }   serde_json = "1.0"   [dev-dependencies]   burn = { version = "0.20.1", features = ["test"] }  
Documentation
Burn Framework: https://burn.dev/
Burn Book: https://burn.dev/book/
Burn Examples: https://github.com/tracel-ai/burn/tree/main/examples
Rust Book: https://doc.rust-lang.org/book/
Transformers Paper: "Attention Is All You Need"
