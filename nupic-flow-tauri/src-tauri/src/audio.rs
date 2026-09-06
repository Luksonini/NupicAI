use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use serde::Serialize;
use std::io::Cursor;
#[cfg(target_os = "linux")]
use std::io::{BufReader, Read};
#[cfg(target_os = "linux")]
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(target_os = "linux")]
use std::thread::JoinHandle;
use std::time::Instant;

const OUTPUT_RATE: u32 = 16_000;
const MAX_SECONDS: usize = 120;

#[derive(Debug, Clone, Serialize)]
pub struct RecordingStatus {
    pub recording: bool,
    pub level: f32,
    pub elapsed_ms: u64,
}

pub struct AudioRecorder {
    backend: Option<CaptureBackend>,
    samples: Arc<Mutex<Vec<f32>>>,
    level_bits: Arc<AtomicU32>,
    sample_rate: u32,
    started_at: Option<Instant>,
}

enum CaptureBackend {
    Microphone(Stream),
    #[cfg(target_os = "linux")]
    SystemAudio {
        child: Child,
        reader: JoinHandle<()>,
    },
}

impl Default for AudioRecorder {
    fn default() -> Self {
        Self {
            backend: None,
            samples: Arc::new(Mutex::new(Vec::new())),
            level_bits: Arc::new(AtomicU32::new(0)),
            sample_rate: OUTPUT_RATE,
            started_at: None,
        }
    }
}

impl AudioRecorder {
    pub fn devices() -> Result<Vec<String>, String> {
        let mut names = cpal::default_host()
            .input_devices()
            .map_err(|error| format!("Nie można odczytać mikrofonów: {error}"))?
            .filter_map(|device| device.name().ok())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        Ok(names)
    }

    pub fn start(&mut self, input_source: &str, preferred_device: &str) -> Result<(), String> {
        if self.backend.is_some() {
            return Ok(());
        }
        if input_source == "system" {
            return self.start_system_audio();
        }
        self.start_microphone(preferred_device)
    }

    fn start_microphone(&mut self, preferred_device: &str) -> Result<(), String> {
        let host = cpal::default_host();
        let device = select_device(&host, preferred_device)?;
        let supported = device
            .default_input_config()
            .map_err(|error| format!("Mikrofon nie udostępnia konfiguracji wejścia: {error}"))?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let channels = usize::from(config.channels);
        let sample_rate = config.sample_rate.0;
        let max_samples = sample_rate as usize * MAX_SECONDS;
        let samples = Arc::new(Mutex::new(Vec::with_capacity(sample_rate as usize * 12)));
        let level_bits = Arc::new(AtomicU32::new(0));
        let stream = build_stream(
            &device,
            &config,
            sample_format,
            Arc::clone(&samples),
            Arc::clone(&level_bits),
            channels,
            max_samples,
        )?;
        stream
            .play()
            .map_err(|error| format!("Nie można uruchomić mikrofonu: {error}"))?;
        self.backend = Some(CaptureBackend::Microphone(stream));
        self.samples = samples;
        self.level_bits = level_bits;
        self.sample_rate = sample_rate;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn start_system_audio(&mut self) -> Result<(), String> {
        let sink_output = Command::new("pactl")
            .arg("get-default-sink")
            .output()
            .map_err(|_| {
                "Dźwięk systemu wymaga pakietu pulseaudio-utils (pactl/parec)".to_string()
            })?;
        if !sink_output.status.success() {
            return Err("Nie można odczytać domyślnego wyjścia PipeWire".into());
        }
        let sink = String::from_utf8_lossy(&sink_output.stdout)
            .trim()
            .to_string();
        if sink.is_empty() {
            return Err("Nie wykryto domyślnego wyjścia audio".into());
        }
        let monitor = format!("{sink}.monitor");
        let mut child = Command::new("parec")
            .arg(format!("--device={monitor}"))
            .arg("--format=s16le")
            .arg(format!("--rate={OUTPUT_RATE}"))
            .arg("--channels=1")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| "Nie można uruchomić przechwytywania dźwięku systemu".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Nie można odczytać dźwięku systemu".to_string())?;
        let samples = Arc::new(Mutex::new(Vec::with_capacity(OUTPUT_RATE as usize * 12)));
        let level_bits = Arc::new(AtomicU32::new(0));
        let reader_samples = Arc::clone(&samples);
        let reader_level = Arc::clone(&level_bits);
        let reader = std::thread::spawn(move || {
            let mut input = BufReader::new(stdout);
            let mut bytes = [0_u8; 4096];
            while input.read_exact(&mut bytes).is_ok() {
                push_pcm16(
                    &bytes,
                    OUTPUT_RATE as usize * MAX_SECONDS,
                    &reader_samples,
                    &reader_level,
                );
            }
        });
        self.backend = Some(CaptureBackend::SystemAudio { child, reader });
        self.samples = samples;
        self.level_bits = level_bits;
        self.sample_rate = OUTPUT_RATE;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn start_system_audio(&mut self) -> Result<(), String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "Nie wykryto domyślnego wyjścia audio".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|error| format!("Wyjście audio nie udostępnia konfiguracji: {error}"))?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let channels = usize::from(config.channels);
        let sample_rate = config.sample_rate.0;
        let max_samples = sample_rate as usize * MAX_SECONDS;
        let samples = Arc::new(Mutex::new(Vec::with_capacity(sample_rate as usize * 12)));
        let level_bits = Arc::new(AtomicU32::new(0));
        let stream = build_stream(
            &device,
            &config,
            sample_format,
            Arc::clone(&samples),
            Arc::clone(&level_bits),
            channels,
            max_samples,
        )?;
        stream
            .play()
            .map_err(|error| format!("Nie można uruchomić WASAPI loopback: {error}"))?;
        self.backend = Some(CaptureBackend::Microphone(stream));
        self.samples = samples;
        self.level_bits = level_bits;
        self.sample_rate = sample_rate;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn start_system_audio(&mut self) -> Result<(), String> {
        Err("Dźwięk systemu nie jest jeszcze dostępny na tym systemie".into())
    }

    pub fn stop_wav(&mut self) -> Result<Vec<u8>, String> {
        if let Some(backend) = self.backend.take() {
            match backend {
                CaptureBackend::Microphone(stream) => drop(stream),
                #[cfg(target_os = "linux")]
                CaptureBackend::SystemAudio { mut child, reader } => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = reader.join();
                }
            }
        }
        self.started_at = None;
        self.level_bits.store(0, Ordering::Relaxed);
        let input = self
            .samples
            .lock()
            .map_err(|_| "Błąd bufora mikrofonu")?
            .clone();
        if input.len() < (self.sample_rate as usize / 4) {
            return Err("Nagranie jest zbyt krótkie".into());
        }
        let mono = resample_linear(&input, self.sample_rate, OUTPUT_RATE);
        let mut cursor = Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: OUTPUT_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer =
                hound::WavWriter::new(&mut cursor, spec).map_err(|error| error.to_string())?;
            for sample in mono {
                writer
                    .write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                    .map_err(|error| error.to_string())?;
            }
            writer.finalize().map_err(|error| error.to_string())?;
        }
        Ok(cursor.into_inner())
    }

    pub fn status(&self) -> RecordingStatus {
        RecordingStatus {
            recording: self.backend.is_some(),
            level: f32::from_bits(self.level_bits.load(Ordering::Relaxed)).clamp(0.0, 1.0),
            elapsed_ms: self
                .started_at
                .map(|started| started.elapsed().as_millis() as u64)
                .unwrap_or(0),
        }
    }
}

fn select_device(host: &cpal::Host, preferred: &str) -> Result<Device, String> {
    if !preferred.trim().is_empty() {
        let mut devices = host.input_devices().map_err(|error| error.to_string())?;
        if let Some(device) =
            devices.find(|device| device.name().ok().as_deref() == Some(preferred))
        {
            return Ok(device);
        }
    }
    host.default_input_device()
        .ok_or_else(|| "Nie wykryto domyślnego mikrofonu".into())
}

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    samples: Arc<Mutex<Vec<f32>>>,
    level: Arc<AtomicU32>,
    channels: usize,
    max_samples: usize,
) -> Result<Stream, String> {
    let error_callback = |error| eprintln!("NupicAI microphone stream error: {error}");
    let result = match format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| {
                push_frames(data, channels, max_samples, &samples, &level, |value| value)
            },
            error_callback,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                push_frames(data, channels, max_samples, &samples, &level, |value| {
                    value as f32 / i16::MAX as f32
                })
            },
            error_callback,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _| {
                push_frames(data, channels, max_samples, &samples, &level, |value| {
                    value as f32 / 32767.5 - 1.0
                })
            },
            error_callback,
            None,
        ),
        _ => return Err(format!("Nieobsługiwany format mikrofonu: {format:?}")),
    };
    result.map_err(|error| format!("Nie można otworzyć mikrofonu: {error}"))
}

fn push_frames<T: Copy>(
    data: &[T],
    channels: usize,
    max_samples: usize,
    samples: &Arc<Mutex<Vec<f32>>>,
    level: &Arc<AtomicU32>,
    convert: impl Fn(T) -> f32,
) {
    let mut peak = 0.0_f32;
    if let Ok(mut output) = samples.lock() {
        for frame in data.chunks(channels.max(1)) {
            if output.len() >= max_samples {
                break;
            }
            let mono = frame.iter().copied().map(&convert).sum::<f32>() / frame.len().max(1) as f32;
            output.push(mono);
            peak = peak.max(mono.abs());
        }
    }
    let visual_level = peak.sqrt().clamp(0.0, 1.0);
    level.store(visual_level.to_bits(), Ordering::Relaxed);
}

#[cfg(target_os = "linux")]
fn push_pcm16(
    bytes: &[u8],
    max_samples: usize,
    samples: &Arc<Mutex<Vec<f32>>>,
    level: &Arc<AtomicU32>,
) {
    let mut peak = 0.0_f32;
    if let Ok(mut output) = samples.lock() {
        let room = max_samples.saturating_sub(output.len());
        let (pairs, _) = bytes.as_chunks::<2>();
        for pair in pairs.iter().take(room) {
            let sample = i16::from_le_bytes([pair[0], pair[1]]) as f32 / i16::MAX as f32;
            output.push(sample);
            peak = peak.max(sample.abs());
        }
    }
    level.store(peak.sqrt().clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

fn resample_linear(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate {
        return input.to_vec();
    }
    let output_len = input.len().saturating_mul(output_rate as usize) / input_rate as usize;
    let ratio = input_rate as f64 / output_rate as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * ratio;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = (position - left as f64) as f32;
            input[left] * (1.0 - fraction) + input[right] * fraction
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::resample_linear;

    #[test]
    fn resampling_preserves_duration() {
        let input = vec![0.25; 48_000];
        let output = resample_linear(&input, 48_000, 16_000);
        assert_eq!(output.len(), 16_000);
        assert!((output[100] - 0.25).abs() < 0.0001);
    }
}
