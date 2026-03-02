#![recursion_limit = "256"]
// src/main.rs
use burn::backend::{Autodiff};
use burn::backend::wgpu::{WgpuDevice, Wgpu};
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
    Train {
        /// Optional path to resume training from a checkpoint
        #[arg(long)]
        model_path: Option<String>,
    },
    /// Ask a question about a document interactively
    Infer {
        /// Path to the .docx document
        #[arg(long)]
        doc_path: String,

        /// Path to the trained model weights
        #[arg(long)]
        model_path: String,
    },
}

fn main() {
    let args = Args::parse();
    // Initialize a WGPU device. You can control the underlying graphics API
    // via the `WGPU_BACKEND` environment variable (e.g. "vulkan", "dx12", "metal").
    let device = WgpuDevice::default();

    // Optionally initialize the runtime to a specific graphics API. If you
    // want a specific API, uncomment and change the line below. The project
    // documentation suggests setting `WGPU_BACKEND` in the environment instead.
    // burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(&device, Default::default());

    match args.action {
        Action::Train { model_path } => {
            training::run_training::<MyAutodiffBackend>(device.clone(), model_path);
        }
        Action::Infer { doc_path, model_path } => {
            if let Err(e) = qa_inference::run_inference::<MyBackend>(doc_path, String::new(), model_path, device.clone()) {
                eprintln!("Inference failed: {}", e);
            }
        }
    }
}