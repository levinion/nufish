use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr, RandomExt};

pub fn randn2(row: usize, col: usize) -> Array2<f64> {
    Array2::random((row, col), rand_distr::Normal::new(0., 1.).unwrap())
}

pub fn randn1(col: usize) -> Array1<f64> {
    Array1::random(col, rand_distr::Normal::new(0., 1.).unwrap())
}

pub fn zeros(row: usize, col: usize) -> Array2<f64> {
    Array2::zeros((row, col))
}

pub fn cmp_f64(a: &&f64, b: &&f64) -> std::cmp::Ordering {
    match a.partial_cmp(b) {
        Some(ord) => ord,
        None => match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, _) => std::cmp::Ordering::Greater,
            (_, true) => std::cmp::Ordering::Less,
            (_, _) => std::cmp::Ordering::Equal, // should never happen
        },
    }
}

pub fn cmp_enumerate_f64(a: &(usize, &f64), b: &(usize, &f64)) -> std::cmp::Ordering {
    match a.1.partial_cmp(b.1) {
        Some(ord) => ord,
        None => match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, _) => std::cmp::Ordering::Greater,
            (_, true) => std::cmp::Ordering::Less,
            (_, _) => std::cmp::Ordering::Equal, // should never happen
        },
    }
}
