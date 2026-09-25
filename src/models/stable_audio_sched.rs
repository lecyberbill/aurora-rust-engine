// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Cosine DPM-Solver++ (EDM, SDE) scheduler for Stable Audio

//! Port of `diffusers.schedulers.CosineDPMSolverMultistepScheduler` (EDM preconditioning,
//! `v_prediction`, SDE DPM-Solver++ order-2, `solver_type="midpoint"`, `final_sigmas_type="zero"`).

use candle_core::Tensor;

const SIGMA_MIN: f64 = 0.3;
const SIGMA_MAX: f64 = 500.0;
const SIGMA_DATA: f64 = 1.0;
const SOLVER_ORDER: usize = 2;

pub struct CosineDpmScheduler {
    /// `[n+1]` sigma schedule (`sigmas[n] = 0`).
    pub sigmas: Vec<f64>,
    /// `[n]` preconditioned timesteps passed to the model.
    pub timesteps: Vec<f64>,
    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: usize,
    pub order: usize,
}

impl CosineDpmScheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let n = num_inference_steps;
        let ln_min = SIGMA_MIN.ln();
        let ln_max = SIGMA_MAX.ln();
        let mut sigmas: Vec<f64> = (0..n)
            .map(|i| {
                let r = i as f64 / (n.max(2) - 1) as f64;
                (ln_min + r * (ln_max - ln_min)).exp()
            })
            .collect();
        sigmas.reverse(); // descending: sigma_max .. sigma_min
        let timesteps: Vec<f64> = sigmas.iter().map(|s| s.atan() / std::f64::consts::PI * 2.0).collect();
        sigmas.push(0.0); // final_sigmas_type = "zero"
        Self {
            sigmas,
            timesteps,
            model_outputs: vec![None; SOLVER_ORDER],
            lower_order_nums: 0,
            order: 1,
        }
    }

    /// `sqrt(sigma_max^2 + 1)`.
    pub fn init_noise_sigma(&self) -> f64 {
        (SIGMA_MAX * SIGMA_MAX + 1.0).sqrt()
    }

    fn c_in(sigma: f64) -> f64 {
        1.0 / (SIGMA_DATA * SIGMA_DATA + sigma * sigma).sqrt()
    }

    /// `scale_model_input`: `c_in(sigma_i) * sample`.
    pub fn scale_input(&self, sample: &Tensor, step: usize) -> candle_core::Result<Tensor> {
        sample.affine(Self::c_in(self.sigmas[step]), 0.0)
    }

    /// `precondition_outputs` (v_prediction): `c_skip*sample + c_out*model_output`.
    fn precondition_outputs(&self, sample: &Tensor, model_output: &Tensor, sigma: f64) -> candle_core::Result<Tensor> {
        let c_skip = SIGMA_DATA * SIGMA_DATA / (sigma * sigma + SIGMA_DATA * SIGMA_DATA);
        let c_out = -sigma * SIGMA_DATA / (sigma * sigma + SIGMA_DATA * SIGMA_DATA).sqrt();
        Ok((sample.affine(c_skip, 0.0)? + model_output.affine(c_out, 0.0)?)?)
    }

    fn first_order(&self, x0: &Tensor, sample: &Tensor, step: usize, noise: &Tensor) -> candle_core::Result<Tensor> {
        let sigma_s = self.sigmas[step];
        let sigma_t = self.sigmas[step + 1];
        let exp_mh = sigma_t / sigma_s; // exp(-h)
        let a = exp_mh * exp_mh; // sigma_t/sigma_s * exp(-h)
        let b = 1.0 - exp_mh * exp_mh;
        let c = sigma_t * b.sqrt();
        Ok(((sample.affine(a, 0.0)? + x0.affine(b, 0.0)?)? + noise.affine(c, 0.0)?)?)
    }

    fn second_order(
        &self,
        m0: &Tensor,
        m1: &Tensor,
        sample: &Tensor,
        step: usize,
        noise: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let sigma_t = self.sigmas[step + 1];
        let sigma_s0 = self.sigmas[step];
        let sigma_s1 = self.sigmas[step - 1];
        let h = (sigma_s0 / sigma_t).ln();
        let h0 = (sigma_s1 / sigma_s0).ln();
        let r0 = h0 / h;
        let exp_mh = sigma_t / sigma_s0;
        let one_m = 1.0 - exp_mh * exp_mh;
        let d1 = (m0 - m1)?.affine(1.0 / r0, 0.0)?;
        let a = exp_mh * exp_mh;
        let b = one_m;
        let d1c = 0.5 * one_m;
        let c = sigma_t * one_m.sqrt();
        let sum = (sample.affine(a, 0.0)? + m0.affine(b, 0.0)?)?;
        let sum = (sum + d1.affine(d1c, 0.0)?)?;
        Ok((&sum + &noise.affine(c, 0.0)?)?)
    }

    /// One SDE DPM-Solver++ step. `model_output` = raw DiT output, `sample` = current EDM-space sample,
    /// `noise` = standard-normal `[B,C,L]`, `step` = index `i`.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        step: usize,
        noise: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let n = self.timesteps.len();
        let sigma = self.sigmas[step];
        let x0 = self.precondition_outputs(sample, model_output, sigma)?;

        for i in 0..SOLVER_ORDER - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
        }
        self.model_outputs[SOLVER_ORDER - 1] = Some(x0.clone());

        let lower_order_final = step == n - 1; // final_sigmas_type == "zero"
        let prev = if SOLVER_ORDER == 1
            || self.lower_order_nums < 1
            || lower_order_final
        {
            self.first_order(&x0, sample, step, noise)?
        } else {
            let m0 = self.model_outputs[SOLVER_ORDER - 1].as_ref().unwrap();
            let m1 = self.model_outputs[SOLVER_ORDER - 2].as_ref().unwrap();
            self.second_order(m0, m1, sample, step, noise)?
        };

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        Ok(prev)
    }
}
