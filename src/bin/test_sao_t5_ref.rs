// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: T5 block0 internals validation

use aurora_rust_engine::models::T5Encoder;
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

fn load_f32(st: &SafeTensors, name: &str, dev: &Device) -> anyhow::Result<Tensor> {
    let t = st.tensor(name)?;
    let shape: Vec<usize> = t.shape().to_vec();
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let bytes = std::fs::read("outputs/audio_ref/sao_ref.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let ids_f = load_f32(&st, "input_ids", &dev)?;
    let am_f = load_f32(&st, "attention_mask", &dev)?;
    let (b, l) = ids_f.dims2()?;
    let ids: Vec<u32> = ids_f.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
    let ids = Tensor::from_vec(ids, (b, l), &dev)?;
    let am = am_f.to_dtype(DType::U32)?;

    let blk = std::fs::read("outputs/audio_ref/sao_t5_block0.safetensors")?;
    let bst = SafeTensors::deserialize(&blk)?;
    let r = |n: &str| load_f32(&bst, n, &dev).unwrap();
    let d = |a: &Tensor, r: &Tensor| (a - r).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();

    let t5 = T5Encoder::from_safetensors(
        "G:/models/Audio/stable-audio-open-models/text_encoder/model.safetensors",
        &dev,
        DType::F32,
    )?;
    let (emb, ln0, a, r1, ln1, ff, out) = t5.debug_block0(&ids, &am)?;
    println!("emb   = {:.3e}", d(&emb, &r("emb")));
    println!("ln0   = {:.3e}", d(&ln0, &r("ln0")));
    println!("attn  = {:.3e}", d(&a, &r("attn")));
    println!("resid = {:.3e}", d(&r1, &r("resid1")));
    println!("ln1   = {:.3e}", d(&ln1, &r("ln1")));
    println!("ff    = {:.3e}", d(&ff, &r("ff")));
    println!("out   = {:.3e}", d(&out, &r("out")));
    let ra = r("attn");
    let ma = a.abs()?.max_all()?.to_scalar::<f32>()?;
    let mr = ra.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("|attn| rust max = {:.4}  ref max = {:.4}", ma, mr);

    // Position bias comparison.
    let bias_r = t5.debug_bias(l)?;
    let ref_bias = {
        let pb = std::fs::read("outputs/audio_ref/sao_t5_bias.safetensors").ok();
        pb.map(|b| {
            let bs = SafeTensors::deserialize(&b).unwrap();
            load_f32(&bs, "bias", &dev).unwrap()
        })
    };
    if let Some(rb) = ref_bias {
        let db = (&bias_r - &rb)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("pos_bias max|diff| = {:.3e}  shapes {:?} vs {:?}", db, bias_r.dims(), rb.dims());
    }

    let qkv = std::fs::read("outputs/audio_ref/sao_t5_qkv.safetensors")?;
    let qst = SafeTensors::deserialize(&qkv)?;
    let (q, k, v, aw, o) = t5.debug_attn0(&ids, &am)?;
    let ref_aw = load_f32(&qst, "aw", &dev)?;
    println!("q  max|diff| = {:.3e}", d(&q, &load_f32(&qst, "q", &dev)?));
    println!("k  max|diff| = {:.3e}", d(&k, &load_f32(&qst, "k", &dev)?));
    println!("v  max|diff| = {:.3e}", d(&v, &load_f32(&qst, "v", &dev)?));
    println!("o  max|diff| = {:.3e}", d(&o, &load_f32(&qst, "o", &dev)?));

    // Restrict to real query AND real key positions.
    let amf = am.to_dtype(DType::F32)?;
    let mm = amf.reshape((b, l, 1))?.broadcast_mul(&amf.reshape((b, 1, l))?)?.reshape((b, 1, l, l))?;
    let dmask = |a: &Tensor, r: &Tensor| -> f32 {
        (a - r).unwrap().abs().unwrap().broadcast_mul(&mm).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
    };
    println!("aw real max|diff| = {:.3e}  (all {:.3e})", dmask(&aw, &ref_aw), d(&aw, &ref_aw));
    let ref_o = load_f32(&qst, "o", &dev)?;
    let am3 = amf.reshape((b, l, 1))?;
    let dq = |a: &Tensor, r: &Tensor| -> f32 {
        (a - r).unwrap().abs().unwrap().broadcast_mul(&am3).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
    };
    println!("o real max|diff| = {:.3e}", dq(&o, &ref_o));
    let rs = t5.debug_scores0(&ids, &am)?;
    let ref_scores = load_f32(&qst, "scores", &dev)?;
    // real positions only (both query and key real)
    let sdiff = (&rs - &ref_scores)?.abs()?.broadcast_mul(&mm)?;
    println!("scores real max|diff| = {:.3e}  mean = {:.3e}", sdiff.max_all()?.to_scalar::<f32>()?, sdiff.mean_all()?.to_scalar::<f32>()?);
    let rs_real = rs.broadcast_mul(&mm)?;
    let rmax = rs_real.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("rust scores real max abs = {:.3e}", rmax);
    // Rust aw real mean diff
    let awd = (&aw - &ref_aw)?.abs()?.broadcast_mul(&mm)?;
    println!("aw real mean|diff| = {:.3e}", awd.mean_all()?.to_scalar::<f32>()?);
    Ok(())
}
