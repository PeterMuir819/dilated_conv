mod dsp;

const SAMPLE_RATE: u32 = 48000;

fn main() {
    impulse_test();
    sine_test();
}

fn impulse_test() {
    let mut input = vec![0.0; 100];
    input[0] = 0.1;

    let kernel = vec![0.5, 0.6, 0.7, 0.8];

    let mut signal = input.clone();
    for d in [1, 2, 4, 8] {
        signal = dsp::layer(signal.as_slice(), &kernel, d);
        let mut count = 0;
        for smpl in signal.iter() {
            if *smpl != 0.0 {
                count += 1;
            }
        }
        println!("Dilation {}: {} non-zero samples", d, count);
    }

    let result = dsp::network(&input, &kernel, &[1, 2, 4, 8]);
    assert_eq!(result, signal);

    println!("{:?}", signal);
}

fn sine_test() {
    let wave = dsp::sine_wave(492.0, SAMPLE_RATE, 2.0);
    dsp::write_wav("sine.wav", &wave, SAMPLE_RATE);

    let dist5 = dsp::distortion(&wave, 5.0);
    dsp::write_wav("dist5.wav", &dist5, SAMPLE_RATE);

    let dist20 = dsp::distortion(&wave, 20.0);
    dsp::write_wav("dist20.wav", &dist20, SAMPLE_RATE);

    println!("{} samples, first 10: {:?}", wave.len(), &wave[..10]);
}
