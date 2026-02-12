# Requirements
    Needs Microsoft Build Tools or Visual Studio with Desktop Development with C++ installed

# Setting up Rust
https://rust-lang.org/tools/install/

# Build and Run Rust
How to Correctly Build and Run:

Once your environment is set up, you should always use cargo commands from the root directory of your project (c:\Users\CPUT\Downloads\2026\Software Engineering\Semester_1\).

To check everything compiles (build the project):

sh
cargo build
To run the training process:

sh
cargo run -- train
To run the inference (question-answering) process:

sh
cargo run -- infer --doc-path "path\to\your\document.docx" --question "What is the date of the ceremony?" --model-path "path\to\your\model_epoch_10"
(Note the -- which separates the arguments for cargo run from the arguments for your program).

By following these steps—correcting your Cargo.toml, ensuring your PATH is set, and using cargo commands—you will be able to compile and run your project as intended.

# Tokenizer
https://huggingface.co/google-bert/bert-base-uncased/tree/main