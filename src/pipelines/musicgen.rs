// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: MusicGen text-to-music pipeline (pure Rust)

//! MusicGen text-to-music: T5 prompt → `enc_to_dec_proj` → AR decoder with the delay-pattern
//! generation over 4 EnCodec codebooks → EnCodec 32 kHz decode.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use std::path::Path;

use crate::audio::{encodec::EncodecDecoder, WavAudio};
use crate::models::{MusicgenDecoder, T5Encoder};

const N_CODEBOOKS: usize = 4;
const PAD: u32 = 2048;

pub struct MusicgenPipeline {
    pub t5: T5Encoder,
    pub decoder: MusicgenDecoder,
    pub encodec: EncodecDecoder,
    pub enc_proj: Linear,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub dtype: DType,
}

impl MusicgenPipeline {
    pub fn from_pretrained<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = DType::F32;
        let dir = dir.as_ref();
        let w = dir.join("model.safetensors");
        let t5 = T5Encoder::from_safetensors_prefix(&w, "text_encoder", &device, dtype).context("T5")?;
        let decoder = MusicgenDecoder::from_musicgen(&w, &device, dtype).context("decoder")?;
        let encodec = EncodecDecoder::from_musicgen(&w, &device, dtype).context("encodec")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&w], dtype, &device)? };
        let enc_proj = linear(768, 1024, vb.pp("enc_to_dec_proj"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Ok(Self {
            t5,
            decoder,
            encodec,
            enc_proj,
            tokenizer,
            device,
            dtype,
        })
    }

    fn encode_text(&self, text: &str) -> Result<Tensor> {
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let l = ids.len();
        let mask: Vec<u32> = vec![1; l];
        let ids_t = Tensor::from_vec(ids, (1, l), &self.device)?;
        let am = Tensor::from_vec(mask, (1, l), &self.device)?;
        let hidden = self.t5.forward(&ids_t, &am)?;
        Ok(self.enc_proj.forward(&hidden)?)
    }

    /// `max_length` = number of generated frames + 4 (delay offsets).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        max_frames: usize,
        guidance_scale: f32,
        temperature: f32,
        top_k: usize,
        seed: u64,
    ) -> Result<WavAudio> {
        let cond = self.encode_text(prompt)?; // [1, L, 1024]
        let lmax = max_frames + N_CODEBOOKS;

        // delay pattern (mono, 4 codebooks): pattern[cb,s] = -1 where prediction is valid.
        let mut shifted = vec![-1i64; N_CODEBOOKS * lmax];
        for cb in 0..N_CODEBOOKS {
            shifted[cb * lmax + cb] = PAD as i64; // prompt BOS shifted by cb
        }
        let mut pattern = vec![0i64; N_CODEBOOKS * lmax];
        for cb in 0..N_CODEBOOKS {
            for s in 0..lmax {
                // delay_pattern = tril + triu(diag = lmax-3)
                let is_pad = s <= cb || s >= cb + lmax - 3;
                pattern[cb * lmax + s] = if is_pad { PAD as i64 } else { shifted[cb * lmax + s] };
            }
        }

        // generated sequence starts with column 0 (all pad), grows column by column
        let mut seq: Vec<Vec<i64>> = vec![vec![PAD as i64]; N_CODEBOOKS]; // [4][len]
        let mut rng = crate::audio::SeededRng::new(seed);
        let _ = (temperature, top_k, &mut rng);

        for _s in 1..lmax {
            let len = seq[0].len();
            // build input [1,4,len] where non-(-1) pattern positions are forced
            let mut data = vec![0u32; N_CODEBOOKS * len];
            for cb in 0..N_CODEBOOKS {
                for (j, &v) in seq[cb].iter().enumerate() {
                    let pv = pattern[cb * lmax + j];
                    data[cb * len + j] = if pv == -1 { v as u32 } else { pv as u32 };
                }
            }
            let inp = Tensor::from_vec(data, (1, N_CODEBOOKS, len), &self.device)?;
            // CFG: [cond; zeros]
            let logits = if guidance_scale > 1.0 {
                let cond2 = Tensor::cat(&[&cond, &Tensor::zeros_like(&cond)?], 0)?;
                let inp2 = Tensor::cat(&[&inp, &inp], 0)?;
                let out = self.decoder.forward(&inp2, &cond2)?; // [2,4,len,2048]
                let last = out.narrow(2, len - 1, 1)?.squeeze(2)?; // [2,4,2048]
                let pos = last.narrow(0, 0, 1)?;
                let neg = last.narrow(0, 1, 1)?;
                (&pos + &(&pos - &neg)?.affine(guidance_scale as f64, 0.0)?)? // [1,4,2048]
            } else {
                let out = self.decoder.forward(&inp, &cond)?; // [1,4,len,2048]
                out.narrow(2, len - 1, 1)?.squeeze(2)?
            };
            // greedy argmax per codebook
            let next = logits.argmax(candle_core::D::Minus1)?.squeeze(0)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
            for cb in 0..N_CODEBOOKS {
                seq[cb].push(next[cb] as i64);
            }
        }

        // collect valid columns per codebook: s in [cb+1, lmax-4+cb]
        let t = lmax - 4;
        let mut codes = Vec::new();
        for cb in 0..N_CODEBOOKS {
            let vals: Vec<u32> = (0..t).map(|k| seq[cb][cb + 1 + k] as u32).collect();
            codes.push(Tensor::from_vec(vals, (1, t), &self.device)?);
        }
        let audio = self.encodec.decode(&codes)?; // [1,1,samples]
        wav_from_mono(&audio, self.encodec.sample_rate as u32)
    }
}

fn wav_from_mono(audio: &Tensor, sr: u32) -> Result<WavAudio> {
    let v: Vec<f32> = audio.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let interleaved: Vec<f32> = v.iter().flat_map(|&s| [s.clamp(-1.0, 1.0), s.clamp(-1.0, 1.0)]).collect();
    Ok(WavAudio::new(interleaved, sr, 2))
}
