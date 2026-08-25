use candle_core::{Device, Tensor};
use cotier_server::model::CorticalModel;
use std::path::PathBuf;

#[test]
fn test_load_and_forward_model() -> cotier_server::Result<()> {
    let model_dir = PathBuf::from("../models/cotier-0.5b");
    if !model_dir.join("model.safetensors").exists() {
        println!("model.safetensors does not exist, skipping test");
        return Ok(());
    }

    let device = Device::Cpu;
    let model = CorticalModel::load(&model_dir, &device)?;

    assert_eq!(model.config.hidden_size, 1024);
    assert_eq!(model.config.num_cortical_stacks, 4);

    // Test forward pass with 4 input tokens
    let input_ids = Tensor::from_vec(vec![1u32, 250u32, 3400u32, 2u32], (1, 4), &device)?;
    let logits = model.forward(&input_ids)?;

    assert_eq!(logits.dims3()?, (1, 4, 32000));
    Ok(())
}

#[test]
fn test_decode_step_and_kv_cache() -> cotier_server::Result<()> {
    let model_dir = PathBuf::from("../models/cotier-0.5b");
    if !model_dir.join("model.safetensors").exists() {
        return Ok(());
    }

    let device = Device::Cpu;
    let model = CorticalModel::load(&model_dir, &device)?;

    let mut kv_caches = vec![None; model.config.num_cortical_stacks];
    let token = Tensor::from_vec(vec![1u32], (1, 1), &device)?;

    let (logits, cycles) = model.forward_decode(&token, 0, &mut kv_caches)?;
    assert_eq!(logits.dims3()?, (1, 1, 32000));
    assert!((1..=6).contains(&cycles));

    // Second decode step
    let token2 = Tensor::from_vec(vec![250u32], (1, 1), &device)?;
    let (logits2, cycles2) = model.forward_decode(&token2, 1, &mut kv_caches)?;
    assert_eq!(logits2.dims3()?, (1, 1, 32000));
    assert!((1..=6).contains(&cycles2));

    Ok(())
}
