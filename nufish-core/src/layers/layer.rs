use super::util::{randn2, zeros};
use crate::tensor::Tensor;
use anyhow::Result;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;

pub type LayerResult = Result<Tensor>;

pub trait Layer {
    fn forward(&mut self, p: &Tensor, if_train: bool) -> LayerResult;
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
    fn forward(&mut self, x: &Tensor, _: bool) -> LayerResult {
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
        let lr = self.lr.unwrap_or(0.01);
        self.weights = self.weights.clone() - lr * self.dw.as_ref().unwrap();
        self.offsets = self.offsets.clone() - lr * self.db.as_ref().unwrap();
    }
}

#[derive(Default)]
pub struct Dropout {
    dropout_ratio: f64,
    mask: Option<Array2<bool>>,
}

impl Layer for Dropout {
    fn forward(&mut self, p: &Tensor, if_train: bool) -> LayerResult {
        if !if_train {
            return Ok(p * (1. - self.dropout_ratio));
        }
        self.mask = Some(
            Tensor::random(p.dim(), ndarray_rand::rand_distr::Uniform::new(0., 1.))
                .mapv(|v| v > self.dropout_ratio),
        );
        let r = p
            .iter()
            .zip(self.mask.as_ref().unwrap().iter())
            .map(|(&p, &mask)| if mask { p } else { 0. })
            .collect::<Array1<f64>>()
            .into_shape(p.dim())?;
        Ok(r)
    }

    fn backward(&mut self, p: &Tensor) -> LayerResult {
        let r = p
            .iter()
            .zip(self.mask.as_ref().unwrap().iter())
            .map(|(&p, &mask)| if mask { p } else { 0. })
            .collect::<Array1<f64>>()
            .into_shape(p.dim())?;
        Ok(r)
    }
}

impl Dropout {
    pub fn new(dropout_ratio: f64) -> Self {
        Self {
            dropout_ratio,
            ..Default::default()
        }
    }
}
