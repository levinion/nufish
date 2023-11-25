use crate::tensor::Tensor;

use super::common;
use super::layer::Layer;
use super::layer::LayerResult;
use ndarray::Array1;

#[derive(Default)]
pub struct Relu {
    mask: Option<ndarray::Array2<bool>>,
}

impl Layer for Relu {
    fn forward(&mut self, x: &Tensor) -> LayerResult {
        let output = x.mapv(common::relu);
        let mask = x.mapv(common::relu_mask);
        self.mask = Some(mask);
        Ok(output)
    }

    fn backward(&mut self, dout: &Tensor) -> LayerResult {
        let (shape0, shape1) = (dout.shape()[0], dout.shape()[1]);
        let dout = dout
            .iter()
            .zip(self.mask.as_ref().unwrap().iter())
            .map(|(d, m)| if m.to_owned() { 0. } else { d.to_owned() })
            .collect::<Array1<f64>>()
            .into_shape((shape0, shape1))
            .unwrap();
        Ok(dout)
    }
}

#[derive(Default)]
pub struct Sigmoid {
    output: Option<Tensor>,
}

impl Layer for Sigmoid {
    fn forward(&mut self, x: &Tensor) -> LayerResult {
        let output = x.mapv(common::sigmoid);
        self.output = Some(output.clone());
        Ok(output)
    }

    fn backward(&mut self, dout: &Tensor) -> LayerResult {
        let output = self.output.as_ref().unwrap();
        Ok(dout * (1. - output) * output)
    }
}
