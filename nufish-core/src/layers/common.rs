use std::f64::consts::E;

use crate::tensor::{max, Tensor};

use super::util::cmp_f64;

pub fn relu(x: f64) -> f64 {
    x.max(0.)
}

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn relu_mask(x: f64) -> bool {
    x <= 0.
}

pub fn softmax(x: &Tensor) -> Tensor {
    if x.ndim() == 2 {
        let mut x_t = x.t().to_owned();
        x_t = x_t.clone() - max(&x_t, 0).unwrap();
        let x_t_exp = x_t.mapv(|v| v.exp());
        let y = x_t_exp.clone() / x_t_exp.sum_axis(ndarray::Axis(0));
        return y.t().to_owned();
    }
    let max = x.iter().max_by(cmp_f64).unwrap();
    let exp_x = x.mapv(|a| (a - max).exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}

pub fn cross_entropy_error(y: &Tensor, t: &Tensor) -> f64 {
    let batch_size = y.shape()[0] as f64;
    let delta = 1e-7;
    -(t * (y + delta).mapv(|x| x.log(E))).sum() / batch_size
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::cross_entropy_error;

    #[test]
    fn test_cross_entropy_error() {
        let t = array![0., 0., 1., 0., 0., 0., 0., 0., 0., 0.];
        let t_reshape = t.clone().into_shape((1, t.len())).unwrap();
        let y = array![0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0.];
        let y_reshape = y.clone().into_shape((1, y.len())).unwrap();
        let error = cross_entropy_error(&y_reshape, &t_reshape);
        assert!(error > 0.5 && error < 0.52)
    }
}
