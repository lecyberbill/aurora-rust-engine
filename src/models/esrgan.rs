// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Real-ESRGAN (RRDBNet & SRVGGNetCompact) architecture

use candle_core::{Module, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

/// Configuration for RRDBNet (Residual-in-Residual Dense Block Network)
#[derive(Debug, Clone)]
pub struct RRDBNetConfig {
    pub in_nc: usize,
    pub out_nc: usize,
    pub num_feat: usize,
    pub num_block: usize,
    pub num_grow_ch: usize,
    pub scale: usize,
}

impl Default for RRDBNetConfig {
    fn default() -> Self {
        Self {
            in_nc: 3,
            out_nc: 3,
            num_feat: 64,
            num_block: 23,
            num_grow_ch: 32,
            scale: 4,
        }
    }
}

/// Helper activation LeakyReLU(negative_slope = 0.2)
#[inline]
pub fn leaky_relu(xs: &Tensor, neg_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    let positive = xs.maximum(&zeros)?;
    let negative = xs.minimum(&zeros)?;
    let neg_scaled = (negative * neg_slope)?;
    positive + neg_scaled
}

/// Residual Dense Block (RDB) with 5 densely connected convolutional layers
#[derive(Debug, Clone)]
pub struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
    res_scale: f64,
}

impl ResidualDenseBlock {
    pub fn new(num_feat: usize, num_grow_ch: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(num_feat, num_grow_ch, 3, cfg, vb.pp("conv1"))?;
        let conv2 = conv2d(num_feat + num_grow_ch, num_grow_ch, 3, cfg, vb.pp("conv2"))?;
        let conv3 = conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, cfg, vb.pp("conv3"))?;
        let conv4 = conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, cfg, vb.pp("conv4"))?;
        let conv5 = conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, cfg, vb.pp("conv5"))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            res_scale: 0.2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = leaky_relu(&self.conv1.forward(x)?, 0.2)?;
        let cat1 = Tensor::cat(&[x, &x1], 1)?;

        let x2 = leaky_relu(&self.conv2.forward(&cat1)?, 0.2)?;
        let cat2 = Tensor::cat(&[x, &x1, &x2], 1)?;

        let x3 = leaky_relu(&self.conv3.forward(&cat2)?, 0.2)?;
        let cat3 = Tensor::cat(&[x, &x1, &x2, &x3], 1)?;

        let x4 = leaky_relu(&self.conv4.forward(&cat3)?, 0.2)?;
        let cat4 = Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1)?;

        let x5 = self.conv5.forward(&cat4)?;
        let res = (x5 * self.res_scale)?;
        x + res
    }
}

/// Residual in Residual Dense Block (RRDB) composed of 3 chained RDBs
#[derive(Debug, Clone)]
pub struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
    res_scale: f64,
}

impl RRDB {
    pub fn new(num_feat: usize, num_grow_ch: usize, vb: VarBuilder) -> Result<Self> {
        let rdb1 = ResidualDenseBlock::new(num_feat, num_grow_ch, vb.pp("rdb1"))?;
        let rdb2 = ResidualDenseBlock::new(num_feat, num_grow_ch, vb.pp("rdb2"))?;
        let rdb3 = ResidualDenseBlock::new(num_feat, num_grow_ch, vb.pp("rdb3"))?;

        Ok(Self {
            rdb1,
            rdb2,
            rdb3,
            res_scale: 0.2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.rdb1.forward(x)?;
        let out = self.rdb2.forward(&out)?;
        let out = self.rdb3.forward(&out)?;
        let res = (out * self.res_scale)?;
        x + res
    }
}

/// RRDBNet: Real-ESRGAN & ESRGAN Deep Neural Upscaler
#[derive(Debug, Clone)]
pub struct RRDBNet {
    conv_first: Conv2d,
    body: Vec<RRDB>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Option<Conv2d>,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    scale: usize,
}

impl RRDBNet {
    pub fn new(cfg: &RRDBNetConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv_first = conv2d(cfg.in_nc, cfg.num_feat, 3, conv_cfg, vb.pp("conv_first"))?;

        // Body RRDB blocks
        let body_vb = if vb.pp("body").contains_tensor("0.rdb1.conv1.weight") || vb.pp("body.0").contains_tensor("rdb1.conv1.weight") {
            vb.pp("body")
        } else {
            vb.pp("rrdbs")
        };

        let mut body = Vec::with_capacity(cfg.num_block);
        for i in 0..cfg.num_block {
            body.push(RRDB::new(cfg.num_feat, cfg.num_grow_ch, body_vb.pp(i))?);
        }

        let conv_body = conv2d(cfg.num_feat, cfg.num_feat, 3, conv_cfg, vb.pp("conv_body"))?;

        // Upsampling layers depending on scale (x2, x4, x8)
        let conv_up1 = conv2d(cfg.num_feat, cfg.num_feat, 3, conv_cfg, vb.pp("conv_up1"))?;
        let conv_up2 = if cfg.scale >= 4 {
            Some(conv2d(cfg.num_feat, cfg.num_feat, 3, conv_cfg, vb.pp("conv_up2"))?)
        } else {
            None
        };

        let conv_hr = conv2d(cfg.num_feat, cfg.num_feat, 3, conv_cfg, vb.pp("conv_hr"))?;
        let conv_last = conv2d(cfg.num_feat, cfg.out_nc, 3, conv_cfg, vb.pp("conv_last"))?;

        Ok(Self {
            conv_first,
            body,
            conv_body,
            conv_up1,
            conv_up2,
            conv_hr,
            conv_last,
            scale: cfg.scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = self.conv_first.forward(x)?;
        let mut body_feat = feat.clone();
        for block in &self.body {
            body_feat = block.forward(&body_feat)?;
        }
        let body_feat = self.conv_body.forward(&body_feat)?;
        let mut feat = (feat + body_feat)?;

        if self.scale == 2 {
            let (_, _, h, w) = feat.dims4()?;
            feat = feat.upsample_nearest2d(h * 2, w * 2)?;
            feat = leaky_relu(&self.conv_up1.forward(&feat)?, 0.2)?;
        } else if self.scale == 4 {
            let (_, _, h, w) = feat.dims4()?;
            feat = feat.upsample_nearest2d(h * 2, w * 2)?;
            feat = leaky_relu(&self.conv_up1.forward(&feat)?, 0.2)?;

            let (_, _, h2, w2) = feat.dims4()?;
            feat = feat.upsample_nearest2d(h2 * 2, w2 * 2)?;
            if let Some(ref conv_up2) = self.conv_up2 {
                feat = leaky_relu(&conv_up2.forward(&feat)?, 0.2)?;
            }
        } else if self.scale == 8 {
            let (_, _, h, w) = feat.dims4()?;
            feat = feat.upsample_nearest2d(h * 2, w * 2)?;
            feat = leaky_relu(&self.conv_up1.forward(&feat)?, 0.2)?;

            let (_, _, h2, w2) = feat.dims4()?;
            feat = feat.upsample_nearest2d(h2 * 2, w2 * 2)?;
            if let Some(ref conv_up2) = self.conv_up2 {
                feat = leaky_relu(&conv_up2.forward(&feat)?, 0.2)?;
            }

            let (_, _, h3, w3) = feat.dims4()?;
            feat = feat.upsample_nearest2d(h3 * 2, w3 * 2)?;
            feat = leaky_relu(&self.conv_hr.forward(&feat)?, 0.2)?;
        }

        let out = leaky_relu(&self.conv_hr.forward(&feat)?, 0.2)?;
        self.conv_last.forward(&out)
    }
}

/// Configuration for SRVGGNetCompact (e.g. RealESRGAN_x4plus_anime_6B)
#[derive(Debug, Clone)]
pub struct SRVGGNetCompactConfig {
    pub num_in_ch: usize,
    pub num_out_ch: usize,
    pub num_feat: usize,
    pub num_conv: usize,
    pub upscale: usize,
}

impl Default for SRVGGNetCompactConfig {
    fn default() -> Self {
        Self {
            num_in_ch: 3,
            num_out_ch: 3,
            num_feat: 64,
            num_conv: 16,
            upscale: 4,
        }
    }
}

/// Compact VGG-based Super-Resolution Network
#[derive(Debug, Clone)]
pub struct SRVGGNetCompact {
    body: Vec<Conv2d>,
    conv_last: Conv2d,
    upscale: usize,
}

impl SRVGGNetCompact {
    pub fn new(cfg: &SRVGGNetCompactConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut body = Vec::new();

        // First conv
        let first_conv = conv2d(cfg.num_in_ch, cfg.num_feat, 3, conv_cfg, vb.pp("body.0"))?;
        body.push(first_conv);

        for i in 1..cfg.num_conv {
            let conv = conv2d(
                cfg.num_feat,
                cfg.num_feat,
                3,
                conv_cfg,
                vb.pp(format!("body.{}", i * 2)),
            )?;
            body.push(conv);
        }

        // Upsampling conv: out channels = num_out_ch * upscale^2 for pixel_shuffle
        let up_out_ch = cfg.num_out_ch * cfg.upscale * cfg.upscale;
        let conv_last = conv2d(
            cfg.num_feat,
            up_out_ch,
            3,
            conv_cfg,
            vb.pp(format!("body.{}", cfg.num_conv * 2)),
        )?;

        Ok(Self {
            body,
            conv_last,
            upscale: cfg.upscale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut feat = x.clone();
        for conv in &self.body {
            feat = leaky_relu(&conv.forward(&feat)?, 0.2)?;
        }
        let out = self.conv_last.forward(&feat)?;

        // PixelShuffle
        pixel_shuffle(&out, self.upscale)
    }
}

/// Pure Rust PixelShuffle (Depth-to-Space) operation
pub fn pixel_shuffle(x: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let r = upscale_factor;
    if c % (r * r) != 0 {
        candle_core::bail!("PixelShuffle: channels {} not divisible by r^2 ({})", c, r * r);
    }
    let out_c = c / (r * r);
    let x = x.reshape((b, out_c, r, r, h, w))?;
    let x = x.permute((0, 1, 4, 2, 5, 3))?;
    x.reshape((b, out_c, h * r, w * r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_pixel_shuffle() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 16, 4, 4), DType::F32, &dev)?;
        let out = pixel_shuffle(&x, 4)?;
        assert_eq!(out.dims4()?, (1, 1, 16, 16));
        Ok(())
    }

    #[test]
    fn test_rrdb_block_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let block = ResidualDenseBlock::new(64, 32, vb.pp("test_rdb"))?;
        let x = Tensor::zeros((1, 64, 8, 8), DType::F32, &dev)?;
        let out = block.forward(&x)?;
        assert_eq!(out.dims4()?, (1, 64, 8, 8));
        Ok(())
    }

    #[test]
    fn test_rrdbnet_forward_x4() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let cfg = RRDBNetConfig {
            in_nc: 3,
            out_nc: 3,
            num_feat: 16, // small feat for fast unit test
            num_block: 2,
            num_grow_ch: 8,
            scale: 4,
        };
        let net = RRDBNet::new(&cfg, vb)?;
        let x = Tensor::zeros((1, 3, 16, 16), DType::F32, &dev)?;
        let out = net.forward(&x)?;
        assert_eq!(out.dims4()?, (1, 3, 64, 64));
        Ok(())
    }

    #[test]
    fn test_srvggnet_compact_forward() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let cfg = SRVGGNetCompactConfig {
            num_in_ch: 3,
            num_out_ch: 3,
            num_feat: 16,
            num_conv: 4,
            upscale: 4,
        };
        let net = SRVGGNetCompact::new(&cfg, vb)?;
        let x = Tensor::zeros((1, 3, 16, 16), DType::F32, &dev)?;
        let out = net.forward(&x)?;
        assert_eq!(out.dims4()?, (1, 3, 64, 64));
        Ok(())
    }
}
