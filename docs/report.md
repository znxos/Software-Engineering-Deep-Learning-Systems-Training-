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
cargo run --release -- train // for faster training
cargo run --release -- train --model-path "model_epoch_2" // resume training

# Optional: Force Vulkan if default DX12 is unstable
$env:WGPU_BACKEND="vulkan" 

cargo run --release -- train


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
\

# Training Output
PS C:\Users\User\Downloads\Software-Engineering-Deep-Learning-Systems-Training-> cargo run --release -- train --model-path "model_epoch_2"
   Compiling word-doc-qa v0.1.0 (C:\Users\User\Downloads\Software-Engineering-Deep-Learning-Systems-Training-)
    Finished `release` profile [optimized] target(s) in 1m 53s
     Running `target\release\word-doc-qa.exe train --model-path model_epoch_2`
Resuming training from model_epoch_2...
Loading training data from: "data\\calendar_2024.docx"
Loading training data from: "data\\calendar_2025.docx"
Loading training data from: "data\\calendar_2026.docx"
Dataset sizes — train: 614 | val: 69
Epoch 3 | Train Iter 0/154 | Loss: 0.9252
Epoch 3 | Train Iter 10/154 | Loss: 1.7971
Epoch 3 | Train Iter 20/154 | Loss: 1.0637
Epoch 3 | Train Iter 30/154 | Loss: 0.5464
Epoch 3 | Train Iter 40/154 | Loss: 1.6986
Epoch 3 | Train Iter 50/154 | Loss: 1.2639
Epoch 3 | Train Iter 60/154 | Loss: 0.2557
Epoch 3 | Train Iter 70/154 | Loss: 0.7552
Epoch 3 | Train Iter 80/154 | Loss: 1.3896
Epoch 3 | Train Iter 90/154 | Loss: 0.6276
Epoch 3 | Train Iter 100/154 | Loss: 1.9406
Epoch 3 | Train Iter 110/154 | Loss: 0.5461
Epoch 3 | Train Iter 120/154 | Loss: 1.4829
Epoch 3 | Train Iter 130/154 | Loss: 1.1377
Epoch 3 | Train Iter 140/154 | Loss: 1.9825
Epoch 3 | Train Iter 150/154 | Loss: 0.8384

--- Epoch 3 Validation ---
Avg Loss: 1.4966 | Avg Accuracy: 0.6667