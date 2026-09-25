use std::fs;
use std::io;

pub struct Data {
    pub training_inputs: Vec<[f64; 2]>,
    pub training_outputs: Vec<f64>,
    pub test_inputs: Vec<[f64; 2]>,
}

/// Reads a CSV with header `feature_1,feature_2,label,split`, where `split`
/// is `train` or `test`. Test rows only contribute their inputs.
pub fn get_data(path: &str) -> io::Result<Data> {
    let contents = fs::read_to_string(path)?;

    let mut training_inputs = Vec::new();
    let mut training_outputs = Vec::new();
    let mut test_inputs = Vec::new();

    for line in contents.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let feature_1: f64 = fields[0].parse().expect("feature_1 must be a number");
        let feature_2: f64 = fields[1].parse().expect("feature_2 must be a number");
        let label: f64 = fields[2].parse().expect("label must be a number");

        match fields[3] {
            "train" => {
                training_inputs.push([feature_1, feature_2]);
                training_outputs.push(label);
            }
            "test" => test_inputs.push([feature_1, feature_2]),
            other => panic!("unknown split '{other}', expected 'train' or 'test'"),
        }
    }

    Ok(Data {
        training_inputs,
        training_outputs,
        test_inputs,
    })
}
