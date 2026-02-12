// src/main.rs
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use clap::Parser;

mod data;
mod model;
mod training;
mod qa_inference;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    action: Action,
}

#[derive(Parser, Debug)]
enum Action {
    /// Train the Q&A model
    Train,
    /// Ask a question about a document
    Infer {
        /// Path to the .docx document
        #[arg(long)]
        doc_path: String,

        /// Question to ask
        #[arg(long)]
        question: String,

        /// Path to the trained model weights
        #[arg(long)]
        model_path: String,
    },
}

fn main() {
    let args = Args::parse();
    let device = WgpuDevice::default();

    match args.action {
        Action::Train => {
            training::run_training::<MyAutodiffBackend>(device);
        }
        Action::Infer { doc_path, question, model_path } => {
            // This now calls the complete inference pipeline.
            qa_inference::run_inference::<MyBackend>(doc_path, question, model_path, device);
        }
    }
}
