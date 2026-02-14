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

cargo run -- infer --doc-path "data\calander_2026.docx" --question "When is the 
start of year administrative staff?" --model-path "model_epoch_12"

# Tokenizer
https://huggingface.co/google-bert/bert-base-uncased/tree/main


# Inaccuary Issues

PS C:\Users\CPUT\Downloads\2026\Software Engineering\Semester_1> cargo run -- infer --doc-path "data\calander_2026.docx" --question "When is autumn  graduation?" --model-path "model_epoch_10"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.20s
     Running `target\debug\word-doc-qa.exe infer --doc-path data\calander_2026.docx --question "When is autumn  graduation?" --model-path model_epoch_10`
Loading model from model_epoch_10...
Model loaded successfully.
Processing document: data\calander_2026.docx
Running inference...

Question: When is autumn  graduation?
Answer: april 2026 may 2026 june 2026 july 2026 august 2026 september 2026 october 2026 november 2026 december
PS C:\Users\CPUT\Downloads\2026\Software Engineering\Semester_1> 