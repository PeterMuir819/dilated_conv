pub fn causal_conv(input: &[f32], kernel: &[f32], dilation: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for n in 0..input.len() {
        let mut sum = 0.0;
        for (k, weight) in kernel.iter().enumerate() {
            let input_index = n as isize - (k as isize * dilation as isize);
            if input_index >= 0 && (input_index as usize) < input.len() {
                sum += input[input_index as usize] * weight;
            }
        }
        output.push(sum);
    }
    output
}

pub fn sine_wave(freq: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * seconds) as usize;
    let mut output = Vec::with_capacity(num_samples);

    for n in 0..num_samples {
        let t = n as f32 / sample_rate as f32;
        let sample = 0.3 * (2.0 * std::f32::consts::PI * freq * t).sin();
        output.push(sample);
    }
    output
}

pub fn write_wav(path: &str, samples: &[f32], sample_rate: u32) {
    let format = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, format).unwrap();
    for sample in samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
}

pub fn distortion(samples: &[f32], gain: f32) -> Vec<f32> {
    let mut output = Vec::with_capacity(samples.len());
    for sample in samples {
        let distorted_sample = 0.3 * (sample * gain).tanh();
        output.push(distorted_sample);
    }
    output
}

pub fn layer(samples: &[f32], kernel: &[f32], dilation: usize) -> Vec<f32> {
    let conv = causal_conv(samples, kernel, dilation);
    let mut output = Vec::with_capacity(conv.len());
    for layered in conv {
        let layered_sample = layered.tanh();
        output.push(layered_sample);
    }
    output
}

pub fn network(samples: &[f32], kernel: &[f32], dilations: &[usize]) -> Vec<f32> {
    let mut signal = samples.to_vec();
    for d in dilations {
        signal = layer(signal.as_slice(), kernel, *d);
    }
    signal
}
