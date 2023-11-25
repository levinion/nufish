use ndarray::{array, s, Array2};
use ndarray_csv::Array2Reader;
use nufish_core::layers::active::Relu;
use nufish_core::layers::last_layer::SoftmaxWithLoss;
use nufish_core::layers::layer::Affine;
use nufish_core::network::Network;
use nufish_core::tensor::Tensor;

fn main() {
    let mut net = Network::new()
        .add_layer(Affine::new(4, 8).set_lr(0.01))
        .add_layer(Relu::default())
        .add_layer(Affine::new(8, 16).set_lr(0.01))
        .add_layer(Relu::default())
        .add_layer(Affine::new(16, 3).set_lr(0.01))
        .set_last_layer(SoftmaxWithLoss::default());

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("./examples/iris.csv")
        .unwrap();

    let data: Array2<String> = reader.deserialize_array2_dynamic().unwrap(); // 输入数据

    let inputs = data
        .slice(s![.., 1..=4])
        .mapv(|v| v.parse::<f64>().unwrap());

    let mut targets = Tensor::default((0, 3));

    data.slice(s![.., 5]).for_each(|v| match &v as &str {
        "setosa" => targets.push_row(array![1., 0., 0.].view()).unwrap(),
        "versicolor" => targets.push_row(array![0., 1., 0.].view()).unwrap(),
        "virginica" => targets.push_row(array![0., 0., 1.].view()).unwrap(),
        _ => unreachable!(),
    });

    for i in 0..=10000000 {
        net.gradient(&inputs, &targets).unwrap();
        net.update().unwrap();
        let loss = net.loss(&inputs, &targets).unwrap();
        let accuracy = net.accuracy(&inputs, &targets).unwrap();
        println!("epoch {i}: loss: {loss}, accuracy: {accuracy}")
    }
}
