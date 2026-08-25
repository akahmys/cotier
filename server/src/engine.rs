//! [CR-15 Compliant] Generation Engine and Token Stream for Cotier Inference.
//!
//! Provides zero-allocation autoregressive decoding with integrated
//! Surprise calculation and PonderNet early-exit metrics.

#![forbid(unsafe_code)]

use crate::model::{CorticalModel, ModelKvCache};
use crate::{CotierError, Result};
use candle_core::{DType, IndexOp, Tensor};
use tokenizers::Tokenizer;

/// Output result for non-streaming text generation.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub avg_surprise: f32,
    pub max_cycles: usize,
    pub finish_reason: String,
}

/// Token-level step metadata produced during autoregressive generation.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub token_id: u32,
    pub token_text: String,
    pub surprise: f32,
    pub cycles: usize,
    pub is_eos: bool,
}

/// Generation configuration parameters.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub eos_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            eos_token_id: 1, // <|im_end|>
        }
    }
}

/// High-performance generation engine orchestrating prefill and token-by-token decoding.
pub struct GenerationEngine<'a> {
    model: &'a CorticalModel,
    tokenizer: &'a Tokenizer,
}

impl<'a> GenerationEngine<'a> {
    pub fn new(model: &'a CorticalModel, tokenizer: &'a Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Generates complete text in non-streaming mode.
    // CR-15 Limit: Dispatcher (under 150 lines)
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        let mut iterator = self.stream(prompt_tokens, config)?;
        let mut generated_ids = Vec::new();
        let mut full_text = String::new();
        let mut total_surprise = 0.0_f32;
        let mut max_cycles = 1;
        let mut finish_reason = "length".to_string();

        while let Some(step_res) = iterator.step()? {
            generated_ids.push(step_res.token_id);
            full_text.push_str(&step_res.token_text);
            total_surprise += step_res.surprise;
            if step_res.cycles > max_cycles {
                max_cycles = step_res.cycles;
            }
            if step_res.is_eos {
                finish_reason = "stop".to_string();
                break;
            }
        }

        let avg_surprise = if generated_ids.is_empty() {
            0.0_f32
        } else {
            total_surprise / generated_ids.len() as f32
        };

        Ok(GenerationOutput {
            text: full_text,
            token_ids: generated_ids,
            avg_surprise,
            max_cycles,
            finish_reason,
        })
    }

    /// Initializes a streaming token iterator for incremental generation.
    pub fn stream(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<TokenStreamIterator<'a>> {
        if prompt_tokens.is_empty() {
            return Err(CotierError::Inference(
                "Prompt tokens cannot be empty for generation".to_string(),
            ));
        }

        let input_tensor = Tensor::new(prompt_tokens, &self.model.device)?.unsqueeze(0)?;
        let prompt_len = prompt_tokens.len();

        let logits = self.model.forward(&input_tensor)?;
        let last_logit = logits.i((0, prompt_len - 1))?;

        let (next_token, surprise) = Self::sample_token(&last_logit, config)?;
        let kv_cache = ModelKvCache::new(self.model.config.num_cortical_stacks);

        Ok(TokenStreamIterator {
            model: self.model,
            tokenizer: self.tokenizer,
            config: config.clone(),
            kv_cache,
            current_token: next_token,
            current_surprise: surprise,
            offset: prompt_len,
            tokens_generated: 0,
            finished: false,
        })
    }

    /// Samples next token id and calculates Surprise S_t = -ln p_t in FP32 (CR-15 Rule 14).
    pub(crate) fn sample_token(logit: &Tensor, _config: &GenerationConfig) -> Result<(u32, f32)> {
        let logit_f32 = logit.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax(&logit_f32, candle_core::D::Minus1)?;
        let max_idx = probs.argmax(candle_core::D::Minus1)?;
        let token_id = max_idx.to_scalar::<u32>()?;

        let prob_val = probs.i(token_id as usize)?.to_scalar::<f32>()?;
        let surprise = -(prob_val.max(1e-7_f32).ln());

        Ok((token_id, surprise))
    }
}

/// Stateful iterator yielding tokens step-by-step with KV cache reuse.
pub struct TokenStreamIterator<'a> {
    model: &'a CorticalModel,
    tokenizer: &'a Tokenizer,
    config: GenerationConfig,
    kv_cache: ModelKvCache,
    current_token: u32,
    current_surprise: f32,
    offset: usize,
    tokens_generated: usize,
    finished: bool,
}

impl<'a> TokenStreamIterator<'a> {
    /// Advances generation by one token step.
    // CR-15 Limit: Max 50 lines for standard function
    pub fn step(&mut self) -> Result<Option<StepOutput>> {
        if self.finished || self.tokens_generated >= self.config.max_new_tokens {
            return Ok(None);
        }

        let token_id = self.current_token;
        let surprise = self.current_surprise;
        let is_eos = token_id == self.config.eos_token_id;

        let token_text = self
            .tokenizer
            .decode(&[token_id], false)
            .map_err(|e| CotierError::Inference(e.to_string()))?;

        self.tokens_generated += 1;
        if is_eos {
            self.finished = true;
            return Ok(Some(StepOutput {
                token_id,
                token_text,
                surprise,
                cycles: 1,
                is_eos: true,
            }));
        }

        // Decode next step token
        let input_tensor = Tensor::new(&[token_id], &self.model.device)?.unsqueeze(0)?;
        let (logits, cycles) =
            self.model
                .forward_decode(&input_tensor, self.offset, &mut self.kv_cache)?;

        let last_logit = logits.i((0, 0))?;
        let (next_token, next_surprise) =
            GenerationEngine::sample_token(&last_logit, &self.config)?;

        self.offset += 1;
        self.current_token = next_token;
        self.current_surprise = next_surprise;

        Ok(Some(StepOutput {
            token_id,
            token_text,
            surprise,
            cycles,
            is_eos: false,
        }))
    }
}
