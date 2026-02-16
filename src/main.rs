// src/main.rs
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
#[cfg(feature = "rocm")]
use burn::backend::rocm::{Rocm, RocmDevice};
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;

mod data;
mod model;
mod training;
mod qa_inference;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Device to use: 'cpu' or 'rocm'
    #[arg(long, default_value = "cpu")]
    device: String,

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

    match args.device.as_str() {
        "cpu" => {
            let device = NdArrayDevice::default();
            run_backend::<Autodiff<NdArray>>(args.action, device);
        }
        "rocm" => {
            #[cfg(feature = "rocm")]
            {
                let device = RocmDevice::default();
                run_backend::<Autodiff<Rocm>>(args.action, device);
            }
            #[cfg(not(feature = "rocm"))]
            {
                panic!("ROCm feature is not enabled. Please compile with --features rocm");
            }
        }
        _ => {
            panic!("Invalid device: {}. Supported: cpu, rocm", args.device);
        }
    }
}

fn run_backend<B: AutodiffBackend>(action: Action, device: B::Device) {
    match action {
        Action::Train => {
            training::run_training::<B>(device);
        }
        Action::Infer { doc_path, question, model_path } => {
            if let Err(e) = qa_inference::run_inference::<B::InnerBackend>(doc_path, question, model_path, device) {
                eprintln!("Inference failed: {}", e);
            }
        }
    }
}
