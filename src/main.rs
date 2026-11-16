mod dsp;

fn main() {

    let mut input = vec![0.0; 1000];
    input[0] = 1.0;

    let kernel = vec![0.5, 0.6, 0.7, 0.8];
    let dilation = 2usize;
    let out = dsp::causal_conv(&input, &kernel, dilation);

    let mut signal = input.clone();
    for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        signal = dsp::layer(signal.as_slice(), &kernel, d);
        let mut count = 0;
        for smpl in signal.iter() {
            if *smpl != 0.0 {
                count += 1;
            }
        }
        println!("Dilation {}: {} non-zero samples", d, count);
    }

    let wave = dsp::sine_wave (492.0, 48000, 2.0);
    dsp::write_wav("sine.wav", &wave, 48000);

    let dist5 = dsp::distortion(&wave,5.0);
    dsp::write_wav("dist5.wav", &dist5, 48000);
    
    let dist20 = dsp::distortion(&wave,20.0);
    dsp::write_wav("dist20.wav", &dist20, 48000);

    println!("{:?}", out);
    println!("{} samples, first 10: {:?}", wave.len(), &wave[..10]);
}

