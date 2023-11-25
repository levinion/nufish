use crate::layers::{
    active::{Relu, Sigmoid},
    last_layer::SoftmaxWithLoss,
    layer::{Affine, Dropout},
};

#[derive(Default)]
pub struct LayerBuilder {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    lr: Option<f64>,
    dropout_ratio: Option<f64>,
}

impl LayerBuilder {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            ..Default::default()
        }
    }

    pub fn set_dropout_radio(mut self, dropout_ratio: f64) -> Self {
        self.dropout_ratio = Some(dropout_ratio);
        self
    }

    pub fn set_lr(mut self, lr: f64) -> Self {
        self.lr = Some(lr);
        self
    }

    pub fn relu(&self) -> Relu {
        Relu::default()
    }

    pub fn sigmoid(&self) -> Sigmoid {
        Sigmoid::default()
    }

    pub fn affine_input(&self) -> Affine {
        let lr = self.lr.unwrap_or(0.01);
        Affine::new(self.input_size, self.hidden_size).set_lr(lr)
    }

    pub fn affine_hidden(&self) -> Affine {
        let lr = self.lr.unwrap_or(0.01);
        Affine::new(self.hidden_size, self.hidden_size).set_lr(lr)
    }

    pub fn affine_output(&self) -> Affine {
        let lr = self.lr.unwrap_or(0.01);
        Affine::new(self.hidden_size, self.output_size).set_lr(lr)
    }

    pub fn dropout(&self) -> Dropout {
        let dropout_ratio = self.dropout_ratio.unwrap_or(0.01);
        Dropout::new(dropout_ratio)
    }

    pub fn softmax_with_loss(&self) -> SoftmaxWithLoss {
        SoftmaxWithLoss::default()
    }
}
