use crate::tensor::Tensor;

use super::common;
use super::layer::LayerResult;
use anyhow::Result;

type LossResult = Result<f64>;

pub trait LastLayer {
    fn forward(&mut self, x: &Tensor, t: &Tensor) -> LossResult;
    fn backward(&mut self) -> LayerResult;
}

#[derive(Default)]
pub struct SoftmaxWithLoss {
    y: Option<Tensor>,
    t: Option<Tensor>,
    loss: Option<f64>,
}

impl LastLayer for SoftmaxWithLoss {
    fn forward(&mut self, x: &Tensor, t: &Tensor) -> LossResult {
        self.t = Some(t.clone());
        let y = common::softmax(x);
        self.y = Some(y.clone());
        let loss = common::cross_entropy_error(&y, t);
        self.loss = Some(loss);
        Ok(loss)
    }

    fn backward(&mut self) -> LayerResult {
        let batch_size = self.t.clone().unwrap().shape()[0] as f64;
        Ok((&self.y.clone().unwrap() - &self.t.clone().unwrap()) / batch_size)
    }
}
