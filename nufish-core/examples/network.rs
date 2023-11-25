use ndarray::{array, s, Array2};
use ndarray_csv::Array2Reader;
// use nufish_core::builder::LayerBuilder;
use nufish_core::layers::active::Relu;
use nufish_core::layers::last_layer::SoftmaxWithLoss;
use nufish_core::layers::layer::{Affine, Dropout};
use nufish_core::network::Network;
use nufish_core::tensor::Tensor;

fn main() {
    let mut net = Network::new()
        .add_layer(Affine::new(4, 8).set_lr(0.01))
        .add_layer(Relu::default())
        .add_layer(Affine::new(8, 8).set_lr(0.01))
        .add_layer(Relu::default())
        .add_layer(Dropout::new(0.5))
        .add_layer(Affine::new(8, 3).set_lr(0.01))
        .set_last_layer(SoftmaxWithLoss::default());

    // You can also use the layer builder:
    //
    // let builder = LayerBuilder::new(4, 8, 3)
    //     .set_lr(0.01)
    //     .set_dropout_radio(0.2);
    //
    // let mut net = Network::new()
    //     .add_layer(builder.affine_input())
    //     .add_layer(builder.relu())
    //     .add_layer(builder.affine_hidden())
    //     .add_layer(builder.relu())
    //     .add_layer(builder.dropout())
    //     .add_layer(builder.affine_output())
    //     .set_last_layer(builder.softmax_with_loss());

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

    let train_x = inputs.slice(s![..120, ..]).to_owned();
    let train_t = targets.slice(s![..120, ..]).to_owned();
    let test_x = inputs.slice(s![120.., ..]).to_owned();
    let test_t = targets.slice(s![120.., ..]).to_owned();

    net.train((&train_x, &train_t), (&test_x, &test_t), 100000)
        .unwrap();
}
