mod data;

use rand::Rng;

struct NeuralNetwork {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let weights = vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)];

        Self {
            weights,
            bias: rng.gen_range(0.0..1.0),
            learning_rate: 0.1,
        }
    }

    fn train(&mut self, inputs: Vec<[f64; 2]>, outputs: Vec<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (i, input) in inputs.iter().enumerate() {

                let output = self.predict(input);

                let error = outputs[i] - output;

                let delta = Self::derivative(output);

                for j in 0..self.weights.len() {
                    self.weights[j] += self.learning_rate * error * input[j] * delta;
                }
                self.bias += self.learning_rate * error * delta;
            }
        } 
    }

    fn derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn predict(&self, input: &[f64; 2]) -> f64 {
        let weighted_sum: f64 = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;

        sigmoid(weighted_sum)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/data.csv");
    let d = data::get_data(data_path).unwrap();

    let inputs = d.training_inputs;
    let outputs = d.training_outputs;
    let test_inputs = d.test_inputs;

    let mut neural_net = NeuralNetwork::new();
    neural_net.train(inputs, outputs, 10_000);

    for input in test_inputs.iter() {
        let prediction = neural_net.predict(input);
        println!("input: {:?}, Prediction: {:.1}", input, prediction);
    }
}