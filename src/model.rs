// src/model.rs
use burn::{
    prelude::*,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig,
        Linear, LinearConfig,
    },
    module::Module,
};

#[derive(Module, Debug)]
pub struct QAModel<B: Backend> {
    embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    token_type_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    output_projection: Linear<B>,
    vocab_size: usize,
    d_model: usize,
    max_seq_length: usize,
}

#[derive(Config, Debug)]
pub struct QAModelConfig {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    #[config(default = "0.1")]
    dropout: f64,
    #[config(default = "512")]
    pub max_seq_length: usize,
}

impl QAModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QAModel<B> {
        let transformer_config = TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
            .with_dropout(self.dropout);

        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_embedding = EmbeddingConfig::new(self.max_seq_length, self.d_model).init(device);
        let token_type_embedding = EmbeddingConfig::new(2, self.d_model).init(device); // For segment embeddings (question/context)
        let transformer = transformer_config.init(device);
        let output_projection = LinearConfig::new(self.d_model, 2).init(device); // 2 outputs: start_logit, end_logit

        QAModel {
            embedding,
            pos_embedding,
            token_type_embedding,
            transformer,
            output_projection,
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> QAModel<B> {
    /// Forward pass for the Q&A model.
    ///
    /// # Shapes
    /// - `tokens`: `[batch_size, seq_length]`
    /// - `token_type_ids`: `[batch_size, seq_length]`
    /// - `mask`: `[batch_size, seq_length]` (attention mask)
    /// - `output`: `[batch_size, seq_length, 2]` (start and end logits)
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        token_type_ids: Tensor<B, 2, Int>, // Used to distinguish question/context
        mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length] = tokens.dims();
        let device = &self.embedding.devices()[0];

        // Token embeddings
        let token_embeds = self.embedding.forward(tokens);

        // Positional embeddings
        let positions = Tensor::arange(0..seq_length as i64, device).unsqueeze::<2>(); // [seq_length, 1]
        let positions = positions.expand([batch_size, seq_length]); // [batch_size, seq_length]
        let pos_embeds = self.pos_embedding.forward(positions);

        // Token type embeddings
        let token_type_embeds = self.token_type_embedding.forward(token_type_ids);

        // Combine embeddings (token + position + token type)
        let x = token_embeds + pos_embeds + token_type_embeds;

        // Transformer Encoder
        let input = TransformerEncoderInput::new(x).mask_pad(mask);
        let encoded = self.transformer.forward(input);

        // Output layer
        self.output_projection.forward(encoded)
    }

    /// Set the model to training mode.
    /// The #[derive(Module)] macro handles training mode internally.
    pub fn train(self) -> Self {
        self
    }

    /// Set the model to evaluation mode.
    /// The #[derive(Module)] macro handles eval mode internally.
    pub fn eval(self) -> Self {
        self
    }
}
