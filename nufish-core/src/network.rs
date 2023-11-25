use crate::{
    layers::{
        last_layer::LastLayer,
        layer::{Layer, LayerResult},
    },
    tensor::{argmax, Tensor},
};

#[derive(Default)]
pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    pub last_layer: Option<Box<dyn LastLayer>>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            last_layer: None,
        }
    }

    pub fn add_layer(mut self, layer: impl Layer + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn set_last_layer(mut self, last_layer: impl LastLayer + 'static) -> Self {
        self.last_layer = Some(Box::new(last_layer));
        self
    }

    pub fn predict(&mut self, x: &Tensor) -> LayerResult {
        let mut x = x.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    pub fn loss(&mut self, x: &Tensor, t: &Tensor) -> anyhow::Result<f64> {
        let y = self.predict(x)?;
        self.last_layer.as_mut().unwrap().forward(&y, t)
    }

    pub fn accuracy(&mut self, x: &Tensor, t: &Tensor) -> anyhow::Result<f64> {
        let y = self.predict(x)?;
        let y_index = argmax(&y, 1)?;
        let t_index = argmax(t, 1)?;
        let correct_num = {
            y_index
                .iter()
                .zip(t_index.iter())
                .map(|(y, t)| if y == t { 1. } else { 0. })
                .sum::<f64>()
        };
        let accuracy = correct_num / (x.shape()[0] as f64);
        Ok(accuracy)
    }

    pub fn gradient(&mut self, x: &Tensor, t: &Tensor) -> anyhow::Result<()> {
        // 前向推理
        self.loss(x, t)?;

        // 误差反向传播
        let mut dout = self.last_layer.as_mut().unwrap().backward()?;
        self.layers.iter_mut().rev().for_each(|layer| {
            dout = layer.backward(&dout).unwrap();
        });

        Ok(())
    }

    pub fn update(&mut self) -> anyhow::Result<()> {
        self.layers.iter_mut().for_each(|layer| layer.update());
        Ok(())
    }
}
