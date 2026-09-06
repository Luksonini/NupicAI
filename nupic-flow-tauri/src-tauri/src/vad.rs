use silero::{SampleRate, Session, StreamState};
use std::collections::VecDeque;

const RATE: usize = 16_000;
const FRAME: usize = 512;
const START_THRESHOLD: f32 = 0.55;
const END_THRESHOLD: f32 = 0.35;
const START_FRAMES: usize = 2;
const END_FRAMES: usize = 19;
const PRE_ROLL_FRAMES: usize = 10;
const POST_ROLL_FRAMES: usize = 6;
const MIN_SPEECH_SAMPLES: usize = RATE / 4;
const MAX_UTTERANCE_SAMPLES: usize = RATE * 30;

pub struct VadUpdate {
    pub speech_started: bool,
    pub utterances: Vec<Vec<f32>>,
}

pub struct VoiceActivityDetector {
    session: Session,
    stream: StreamState,
    resampler: StreamingResampler,
    pending: Vec<f32>,
    pre_roll: VecDeque<f32>,
    utterance: Vec<f32>,
    active: bool,
    speech_frames: usize,
    silence_frames: usize,
}

impl VoiceActivityDetector {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            session: Session::bundled().map_err(|error| error.to_string())?,
            stream: StreamState::new(SampleRate::Rate16k),
            resampler: StreamingResampler::default(),
            pending: Vec::new(),
            pre_roll: VecDeque::with_capacity(FRAME * PRE_ROLL_FRAMES),
            utterance: Vec::new(),
            active: false,
            speech_frames: 0,
            silence_frames: 0,
        })
    }

    pub fn push(&mut self, input: &[f32], input_rate: u32) -> Result<VadUpdate, String> {
        let samples = self.resampler.process(input, input_rate);
        self.pending.extend(samples);
        let mut update = VadUpdate {
            speech_started: false,
            utterances: Vec::new(),
        };
        let complete = self.pending.len() / FRAME * FRAME;
        let frames = self.pending[..complete].to_vec();
        self.pending.drain(..complete);
        let (frames, _) = frames.as_chunks::<FRAME>();
        for frame in frames {
            let probability = self
                .session
                .infer_chunk(&mut self.stream, frame)
                .map_err(|error| error.to_string())?;
            if let Some(utterance) = self.process_frame(frame, probability, &mut update) {
                update.utterances.push(utterance);
            }
        }
        Ok(update)
    }

    pub fn finish(&mut self) -> Option<Vec<f32>> {
        if !self.active || self.utterance.len() < MIN_SPEECH_SAMPLES {
            return None;
        }
        self.active = false;
        self.speech_frames = 0;
        self.silence_frames = 0;
        Some(std::mem::take(&mut self.utterance))
    }

    fn process_frame(
        &mut self,
        frame: &[f32],
        probability: f32,
        update: &mut VadUpdate,
    ) -> Option<Vec<f32>> {
        if !self.active {
            self.push_pre_roll(frame);
            self.speech_frames = if probability >= START_THRESHOLD {
                self.speech_frames + 1
            } else {
                0
            };
            if self.speech_frames >= START_FRAMES {
                self.active = true;
                self.silence_frames = 0;
                self.utterance.extend(self.pre_roll.drain(..));
                update.speech_started = true;
            }
            return None;
        }

        self.utterance.extend_from_slice(frame);
        self.silence_frames = if probability <= END_THRESHOLD {
            self.silence_frames + 1
        } else {
            0
        };
        if self.silence_frames >= END_FRAMES {
            let trim = (END_FRAMES - POST_ROLL_FRAMES) * FRAME;
            self.utterance
                .truncate(self.utterance.len().saturating_sub(trim));
            return self.take_utterance();
        }
        if self.utterance.len() >= MAX_UTTERANCE_SAMPLES {
            return self.take_utterance();
        }
        None
    }

    fn take_utterance(&mut self) -> Option<Vec<f32>> {
        self.active = false;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.stream.reset();
        self.pre_roll.clear();
        let utterance = std::mem::take(&mut self.utterance);
        (utterance.len() >= MIN_SPEECH_SAMPLES).then_some(utterance)
    }

    fn push_pre_roll(&mut self, frame: &[f32]) {
        let limit = FRAME * PRE_ROLL_FRAMES;
        self.pre_roll.extend(frame.iter().copied());
        while self.pre_roll.len() > limit {
            self.pre_roll.pop_front();
        }
    }
}

#[derive(Default)]
struct StreamingResampler {
    phase: u64,
    input_rate: u32,
}

impl StreamingResampler {
    fn process(&mut self, input: &[f32], input_rate: u32) -> Vec<f32> {
        if input_rate == RATE as u32 {
            return input.to_vec();
        }
        if self.input_rate != input_rate {
            self.phase = 0;
            self.input_rate = input_rate;
        }
        if input_rate < RATE as u32 {
            return super::audio::resample_linear(input, input_rate, RATE as u32);
        }
        let mut output =
            Vec::with_capacity(input.len().saturating_mul(RATE) / input_rate.max(1) as usize + 1);
        for sample in input {
            self.phase += RATE as u64;
            if self.phase >= input_rate as u64 {
                self.phase -= input_rate as u64;
                output.push(*sample);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::VoiceActivityDetector;

    #[test]
    fn bundled_vad_accepts_streaming_silence() {
        let mut vad = VoiceActivityDetector::new().expect("bundled Silero model");
        let update = vad.push(&vec![0.0; 16_000], 16_000).expect("VAD inference");
        assert!(!update.speech_started);
        assert!(update.utterances.is_empty());
        assert!(vad.finish().is_none());
    }
}
