use super::util::{randn2, zeros};
use crate::tensor::Tensor;
use anyhow::Result;

pub type LayerResult = Result<Tensor>;

pub trait Layer {
    fn forward(&mut self, p: &Tensor) -> LayerResult;
    fn backward(&mut self, p: &Tensor) -> LayerResult;
    fn update(&mut self) {}
}

/// 全连接层，L是上一层神经元数量，C是该层神经元数量，N表示数据量
#[derive(Default)]
pub struct Affine {
    size: usize,
    weights: Tensor,
    offsets: Tensor,
    input: Option<Tensor>,
    dw: Option<Tensor>,
    db: Option<Tensor>,
    lr: Option<f64>,
}

impl Affine {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = 0.01 * randn2(input_size, output_size);
        let offsets = zeros(1, output_size);
        Self {
            weights,
            offsets,
            size: output_size,
            ..Default::default()
        }
    }

    pub fn set_lr(mut self, lr: f64) -> Self {
        self.lr = Some(lr);
        self
    }
}

impl Layer for Affine {
    fn forward(&mut self, x: &Tensor) -> LayerResult {
        self.input = Some(x.clone());
        let x_dot_w = x.dot(&self.weights);
        Ok(x_dot_w + &self.offsets)
    }

    fn backward(&mut self, dout: &Tensor) -> LayerResult {
        let dx = dout.dot(&self.weights.t());
        self.dw = Some(self.input.as_ref().unwrap().t().dot(dout));
        self.db = Some(
            dout.sum_axis(ndarray::Axis(0))
                .into_shape((1, self.size))
                .unwrap(),
        );
        Ok(dx)
    }

    fn update(&mut self) {
        let lr = self.lr.unwrap_or(0.001);
        self.weights = self.weights.clone() - lr * self.dw.as_ref().unwrap();
        self.offsets = self.offsets.clone() - lr * self.db.as_ref().unwrap();
    }
}
