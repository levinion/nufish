use anyhow::anyhow;
use ndarray::Array2;

use crate::layers::util::{cmp_enumerate_f64, cmp_f64};

pub type Tensor = ndarray::Array2<f64>;

pub fn argmax(tensor: &Tensor, axis: usize) -> anyhow::Result<Array2<usize>> {
    let r = match axis {
        0 => {
            let mut r = vec![];
            for column in tensor.columns() {
                let max = column.iter().enumerate().max_by(cmp_enumerate_f64).unwrap();
                r.push(max.0);
            }
            Ok(r)
        }
        1 => {
            let mut r = vec![];
            for row in tensor.rows() {
                let max = row.iter().enumerate().max_by(cmp_enumerate_f64).unwrap();
                r.push(max.0);
            }
            Ok(r)
        }
        _ => Err(anyhow!("axis can not be greater than 2")),
    };
    let long = r.as_ref().unwrap().len();
    r.map(|v| Array2::from_shape_vec((1, long), v).unwrap())
}

pub fn max(tensor: &Tensor, axis: usize) -> anyhow::Result<Tensor> {
    let r = match axis {
        0 => {
            let mut r = vec![];
            for column in tensor.columns() {
                let max = *column.iter().max_by(cmp_f64).unwrap();
                r.push(max);
            }
            Ok(r)
        }
        1 => {
            let mut r = vec![];
            for row in tensor.rows() {
                let max = *row.iter().max_by(cmp_f64).unwrap();
                r.push(max);
            }
            Ok(r)
        }
        _ => Err(anyhow!("axis can not be greater than 2")),
    };
    let long = r.as_ref().unwrap().len();
    r.map(|v| Array2::from_shape_vec((1, long), v).unwrap())
}
