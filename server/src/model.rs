#![forbid(unsafe_code)]

use crate::{CotierError, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let new_k = Tensor::cat(&[prev_k, k], 2)?.contiguous()?;
                let new_v = Tensor::cat(&[prev_v, v], 2)?.contiguous()?;
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
                Ok((new_k, new_v))
            }
            _ => {
                self.k = Some(k.clone());
                self.v = Some(v.clone());
                Ok((k.clone(), v.clone()))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelKvCache {
    pub layers: Vec<LayerKvCache>,
}

impl ModelKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerKvCache::new(); num_layers],
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layer_mut(&mut self, idx: usize) -> Result<&mut LayerKvCache> {
        self.layers.get_mut(idx).ok_or_else(|| {
            CotierError::Inference(format!("Layer KV cache index {} out of bounds", idx))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_num_stacks")]
    pub num_cortical_stacks: usize,
    #[serde(default = "default_max_cycles")]
    pub max_recurrent_cycles: usize,
    #[serde(default = "default_alpha")]
    pub recurrent_alpha: f64,
    #[serde(default = "default_ponder_epsilon")]
    pub ponder_epsilon: f64,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_intermediate_size() -> usize {
    2816
}
fn default_num_heads() -> usize {
    16
}
fn default_head_dim() -> usize {
    64
}
fn default_num_stacks() -> usize {
    4
}
fn default_max_cycles() -> usize {
    6
}
fn default_alpha() -> f64 {
    0.1
}
fn default_ponder_epsilon() -> f64 {
    0.05
}
fn default_vocab_size() -> usize {
    32000
}
fn default_max_positions() -> usize {
    2048
}
fn default_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_attention_heads: default_num_heads(),
            num_key_value_heads: default_num_heads(),
            head_dim: default_head_dim(),
            num_cortical_stacks: default_num_stacks(),
            max_recurrent_cycles: default_max_cycles(),
            recurrent_alpha: default_alpha(),
            ponder_epsilon: default_ponder_epsilon(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_positions(),
            rms_norm_eps: default_eps(),
            rope_theta: default_rope_theta(),
        }
    }
}

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps_t = Tensor::new(self.eps as f32, x.device())?;
        let rsqrt = (variance.broadcast_add(&eps_t)?).sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rsqrt)?.to_dtype(x.dtype())?;
        let out = normed.broadcast_mul(&self.weight)?;
        Ok(out)
    }
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half_dim = dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / (theta.powf((2.0 * i as f64) / (dim as f64)));
            inv_freq.push(freq as f32);
        }

        let mut cos_table = Vec::with_capacity(max_seq_len * dim);
        let mut sin_table = Vec::with_capacity(max_seq_len * dim);

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let val = (pos as f32) * freq;
                cos_table.push(val.cos());
            }
            for &freq in &inv_freq {
                let val = (pos as f32) * freq;
                cos_table.push(val.cos());
            }
            for &freq in &inv_freq {
                let val = (pos as f32) * freq;
                sin_table.push(val.sin());
            }
            for &freq in &inv_freq {
                let val = (pos as f32) * freq;
                sin_table.push(val.sin());
            }
        }

        let cos = Tensor::from_vec(cos_table, (max_seq_len, dim), device)?;
        let sin = Tensor::from_vec(sin_table, (max_seq_len, dim), device)?;

        Ok(Self { cos, sin })
    }

    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        // x: [B, H, L, D]
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;

        // Broadcast to [1, 1, L, D]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x.shape())?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x.shape())?;

        let half_dim = head_dim / 2;
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let neg_x2 = x2.neg()?;
        let rotated = Tensor::cat(&[&neg_x2, &x1], 3)?;

        let out = (x.mul(&cos)? + rotated.mul(&sin)?)?;
        Ok(out)
    }
}

pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let q_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("o_proj"))?;
        let rotary = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        offset: usize,
        kv_cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        let (b, l, _d) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.rotary.apply(&q, offset)?.contiguous()?;
        let k = self.rotary.apply(&k, offset)?.contiguous()?;

        let (k, v) = kv_cache.append(&k, &v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let att = (q.matmul(&k_t)? * scale)?;

        // Apply causal mask if l > 1
        let (_b, _h, _q_len, kv_len) = att.dims4()?;
        let att = if l > 1 {
            let mask = self.causal_mask(l, kv_len, x.device())?;
            att.broadcast_add(&mask)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?.contiguous()?;
        let v_cont = v.contiguous()?;
        let out = att.matmul(&v_cont)?;
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, l, self.num_heads * self.head_dim))?;
        let out = self.o_proj.forward(&out)?;
        Ok(out)
    }

    fn causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mut mask = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                if j > i + (kv_len - q_len) {
                    mask[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?;
        Ok(mask)
    }
}

pub struct SwiGLUFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLUFFN {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let down = self.down_proj.forward(&(gate * up)?)?;
        Ok(down)
    }
}

pub struct LayerVCore {
    attn_norm: RMSNorm,
    attn: CausalSelfAttention,
    ffn_norm: RMSNorm,
    ffn: SwiGLUFFN,
    alpha: f64,
}

impl LayerVCore {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let attn_norm = RMSNorm::new(vb.pp("attn_norm"), config.hidden_size, config.rms_norm_eps)?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), config)?;
        let ffn_norm = RMSNorm::new(vb.pp("ffn_norm"), config.hidden_size, config.rms_norm_eps)?;
        let ffn = SwiGLUFFN::new(vb.pp("ffn"), config)?;
        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
            alpha: config.recurrent_alpha,
        })
    }

    pub fn forward(
        &self,
        z_prev: &Tensor,
        z_l1: &Tensor,
        offset: usize,
        kv_cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        let a_k = self.attn.forward(z_prev, offset, kv_cache)?;
        let sum_z = (z_prev + &a_k)?.broadcast_add(z_l1)?;
        let u_k = self.attn_norm.forward(&sum_z)?;
        let ffn_in = self.ffn_norm.forward(&u_k)?;
        let ffn_out = self.ffn.forward(&ffn_in)?;
        let z_tilde = (z_prev + (ffn_out * self.alpha)?)?;
        let z_k = self.ffn_norm.forward(&z_tilde)?;
        Ok(z_k)
    }
}

pub struct LayerVIHalting {
    linear: Linear,
}

impl LayerVIHalting {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let linear = candle_nn::linear(config.hidden_size, 1, vb)?;
        Ok(Self { linear })
    }

    pub fn compute_h_k(&self, z_k: &Tensor) -> Result<Tensor> {
        let z_f32 = z_k.to_dtype(DType::F32)?;
        let logits = self.linear.forward(&z_f32)?;
        let h_k = candle_nn::ops::sigmoid(&logits)?;
        Ok(h_k)
    }
}

pub struct CorticalStack {
    l1_global: Linear,
    l1_global_norm: RMSNorm,
    l2_lateral: Linear,
    l2_lateral_norm: RMSNorm,
    l4_norm: RMSNorm,
    l5_core: LayerVCore,
    l6_halt: LayerVIHalting,
    max_cycles: usize,
    ponder_eps: f64,
}

impl CorticalStack {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let l1_global = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("l1_global"))?;
        let l1_global_norm = RMSNorm::new(
            vb.pp("l1_global_norm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let l2_lateral =
            linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("l2_lateral"))?;
        let l2_lateral_norm = RMSNorm::new(
            vb.pp("l2_lateral_norm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let l4_norm = RMSNorm::new(vb.pp("l4_norm"), config.hidden_size, config.rms_norm_eps)?;
        let l5_core = LayerVCore::new(vb.pp("l5_core"), config)?;
        let l6_halt = LayerVIHalting::new(vb.pp("l6_halt"), config)?;

        Ok(Self {
            l1_global,
            l1_global_norm,
            l2_lateral,
            l2_lateral_norm,
            l4_norm,
            l5_core,
            l6_halt,
            max_cycles: config.max_recurrent_cycles,
            ponder_eps: config.ponder_epsilon,
        })
    }

    pub fn forward(
        &self,
        z_in: &Tensor,
        offset: usize,
        kv_cache: &mut LayerKvCache,
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        let z_l4 = self.l4_norm.forward(z_in)?;
        let bar_context = z_in.mean_keepdim(1)?;
        let z_l1 = self
            .l1_global_norm
            .forward(&self.l1_global.forward(&bar_context)?)?;

        let lateral_in = z_l4.broadcast_add(&z_l1)?;
        let mut z_k = self
            .l2_lateral_norm
            .forward(&self.l2_lateral.forward(&lateral_in)?)?;

        let target_cycles = if is_prefill { 1 } else { self.max_cycles };
        let mut accum_p = 0.0f32;
        let mut accum_not_halt = 1.0f32;
        let mut cycles_used = 0;

        for step in 0..target_cycles {
            cycles_used = step + 1;
            z_k = self.l5_core.forward(&z_k, &z_l1, offset, kv_cache)?;

            if !is_prefill {
                let h_k_t = self.l6_halt.compute_h_k(&z_k)?;
                let last_token_h = h_k_t.i((0, h_k_t.dim(1)? - 1, 0))?.to_scalar::<f32>()?;
                let p_k = last_token_h * accum_not_halt;
                accum_p += p_k;
                accum_not_halt *= 1.0 - last_token_h;

                if (accum_p >= (1.0 - self.ponder_eps as f32)) || (cycles_used == target_cycles) {
                    break;
                }
            }
        }

        Ok((z_k, cycles_used))
    }
}

pub struct CorticalModel {
    pub config: ModelConfig,
    pub device: Device,
    embed_tokens: Embedding,
    stacks: Vec<CorticalStack>,
    norm: RMSNorm,
    lm_head: Linear,
}

impl CorticalModel {
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let safetensors_path = dir.join("model.safetensors");

        if !safetensors_path.exists() {
            return Err(CotierError::ModelLoad(format!(
                "model.safetensors not found in {:?}",
                dir
            )));
        }

        let config: ModelConfig = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            ModelConfig::default()
        };

        let file_data = std::fs::read(&safetensors_path)?;
        let vb = VarBuilder::from_buffered_safetensors(file_data, DType::F32, device)?;

        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut stacks = Vec::with_capacity(config.num_cortical_stacks);
        for i in 0..config.num_cortical_stacks {
            let stack_vb = vb.pp(format!("stacks.{}", i));
            stacks.push(CorticalStack::new(stack_vb, &config)?);
        }

        let norm = RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            config,
            device: device.clone(),
            embed_tokens,
            stacks,
            norm,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for stack in &self.stacks {
            let mut stack_kv = LayerKvCache::new();
            let (z_final, _) = stack.forward(&hidden, 0, &mut stack_kv, true)?;
            hidden = z_final;
        }

        let normed = self.norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&normed)?;
        Ok(logits)
    }

    pub fn forward_decode(
        &self,
        token_id: &Tensor,
        offset: usize,
        model_kv: &mut ModelKvCache,
    ) -> Result<(Tensor, usize)> {
        let mut hidden = self.embed_tokens.forward(token_id)?;
        let mut max_cycles = 1;

        for (idx, stack) in self.stacks.iter().enumerate() {
            let layer_kv = model_kv.layer_mut(idx)?;
            let (z_final, cycles) = stack.forward(&hidden, offset, layer_kv, false)?;
            hidden = z_final;
            if cycles > max_cycles {
                max_cycles = cycles;
            }
        }

        let normed = self.norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&normed)?;
        Ok((logits, max_cycles))
    }
}
