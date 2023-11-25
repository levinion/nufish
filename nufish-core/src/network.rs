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

    fn predict(&mut self, x: &Tensor, if_train: bool) -> LayerResult {
        let mut x = x.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x, if_train)?;
        }
        Ok(x)
    }

    fn loss(&mut self, x: &Tensor, t: &Tensor, if_train: bool) -> anyhow::Result<f64> {
        let y = self.predict(x, if_train)?;
        self.last_layer.as_mut().unwrap().forward(&y, t)
    }

    fn accuracy(&mut self, x: &Tensor, t: &Tensor, if_train: bool) -> anyhow::Result<f64> {
        let y = self.predict(x, if_train)?;
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

    fn gradient(&mut self, x: &Tensor, t: &Tensor, if_train: bool) -> anyhow::Result<()> {
        // 前向推理
        self.loss(x, t, if_train)?;

        // 误差反向传播
        let mut dout = self.last_layer.as_mut().unwrap().backward()?;
        self.layers.iter_mut().rev().for_each(|layer| {
            dout = layer.backward(&dout).unwrap();
        });

        Ok(())
    }

    fn update(&mut self) -> anyhow::Result<()> {
        self.layers.iter_mut().for_each(|layer| layer.update());
        Ok(())
    }

    pub fn train(
        &mut self,
        train_data: (&Tensor, &Tensor),
        test_data: (&Tensor, &Tensor),
        epoch: usize,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        let (train_x, train_t) = train_data;
        let (test_x, test_t) = test_data;
        let batch_number = train_x.nrows() / batch_size;
        for i in 1..=epoch {
            for index in 0..batch_number {
                let range: Vec<_> = (index * batch_size..(index + 1) * batch_size).collect();
                let train_x = train_x.select(ndarray::Axis(0), &range);
                let train_t = train_t.select(ndarray::Axis(0), &range);
                self.gradient(&train_x, &train_t, true)?;
                self.update()?;
            }
            let train_loss = self.loss(train_x, train_t, true)?;
            let test_loss = self.loss(test_x, test_t, false)?;
            let train_accuracy = self.accuracy(train_x, train_t, true)?;
            let test_accuracy = self.accuracy(test_x, test_t, false)?;
            println!("epoch {i}: \n\ttrain_loss: {train_loss}, \n\ttrain_acc: {train_accuracy}, \n\ttest_loss: {test_loss}, \n\ttest_acc: {test_accuracy}")
        }
        Ok(())
    }
}
