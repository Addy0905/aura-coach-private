// ------------------------------------------------------------
//  AudioAnalyzer – fixed MPM pitch detection + robust API
// ------------------------------------------------------------
export interface AudioFeatures {
  pitch: number;            // Hz (MPM)
  pitchVariation: number;   // 0-100
  volume: number;           // dB
  volumeVariation: number;  // 0-100
  pace: number;             // words/min (external)
  clarity: number;          // 0-100
  energy: number;           // 0-100
  spectralCentroid: number; // bin index (≈ brightness)
  zeroCrossingRate: number; // 0-100
  snr: number;              // dB
  confidence: number;       // 0-100
}

export class AudioAnalyzer {
  // ------------------------------------------------------------------
  //  Private fields
  // ------------------------------------------------------------------
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  private timeData: Float32Array | null = null;
  private freqData: Uint8Array | null = null;
  private floatFreq: Float32Array | null = null;   // for real FFT

  private pitchHistory: number[] = [];
  private volumeHistory: number[] = [];
  private energyHistory: number[] = [];

  private noiseFloor = -60;
  private noiseCalibrated = false;
  private calibrationSamples: number[] = [];

  private frameCount = 0;
  private readonly ANALYSIS_INTERVAL = 2; // analyse every N frames

  // ------------------------------------------------------------------
  //  Constants (tuned for human speech)
  // ------------------------------------------------------------------
  private readonly VOICE_THRESHOLD_DB = -50;
  private readonly MIN_PITCH = 75;
  private readonly MAX_PITCH = 600;
  private readonly MPM_CUTOFF = 0.93;
  private readonly MIN_CONFIDENCE = 0.85;

  // ------------------------------------------------------------------
  //  PUBLIC API
  // ------------------------------------------------------------------
  initialize(stream: MediaStream): void {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 44100,
        latencyHint: 'interactive',
      });

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 4096;               // good low-freq resolution
      this.analyser.smoothingTimeConstant = 0.85;
      this.analyser.minDecibels = -100;
      this.analyser.maxDecibels = -10;

      this.source = this.audioContext.createMediaStreamSource(stream);
      this.source.connect(this.analyser);

      const bufLen = this.analyser.fftSize;
      this.timeData = new Float32Array(bufLen);
      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this.floatFreq = new Float32Array(this.analyser.frequencyBinCount);

      this.resetState();
      console.log('AudioAnalyzer initialized (MPM + HPS)');
    } catch (e) {
      console.error('Audio init error:', e);
      throw new Error('Microphone access denied or unsupported browser');
    }
  }

  getAudioFeatures(): AudioFeatures {
    if (!this.analyser || !this.timeData || !this.freqData || !this.floatFreq) {
      return this.defaultFeatures();
    }

    // ----------------------------------------------------------------
    //  Grab fresh data
    // ----------------------------------------------------------------
    this.analyser.getFloatTimeDomainData(this.timeData);
    this.analyser.getByteFrequencyData(this.freqData);
    this.analyser.getFloatFrequencyData(this.floatFreq);

    const rms = this.rms(this.timeData);
    const volumeDB = this.toDB(rms);

    // ----------------------------------------------------------------
    //  Noise-floor calibration (first ~20 quiet frames)
    // ----------------------------------------------------------------
    if (!this.noiseCalibrated) this.calibrateNoise(volumeDB);

    const snr = this.noiseCalibrated ? volumeDB - this.noiseFloor : 0;

    // ----------------------------------------------------------------
    //  Quick voice gate
    // ----------------------------------------------------------------
    const voiceGate = volumeDB > this.VOICE_THRESHOLD_DB && snr > 3;
    if (!voiceGate) return this.defaultFeatures();

    // ----------------------------------------------------------------
    //  Pitch (MPM) – only every ANALYSIS_INTERVAL frames
    // ----------------------------------------------------------------
    let pitch = 0;
    let confidence = 0;

    this.frameCount++;
    if (this.frameCount % this.ANALYSIS_INTERVAL === 0) {
      const mpm = this.mpmPitch(this.timeData);
      pitch = mpm.pitch;
      confidence = mpm.confidence;

      if (this.isValidPitch(pitch) && confidence > this.MIN_CONFIDENCE) {
        this.pitchHistory.push(pitch);
        if (this.pitchHistory.length > 60) this.pitchHistory.shift();
      }
    } else if (this.pitchHistory.length) {
      pitch = this.pitchHistory[this.pitchHistory.length - 1];
    }

    // ----------------------------------------------------------------
    //  Volume & energy history
    // ----------------------------------------------------------------
    this.volumeHistory.push(volumeDB);
    if (this.volumeHistory.length > 60) this.volumeHistory.shift();

    const energy = this.energy(this.freqData);
    this.energyHistory.push(energy);
    if (this.energyHistory.length > 40) this.energyHistory.shift();

    // ----------------------------------------------------------------
    //  Spectral features
    // ----------------------------------------------------------------
    const centroid = this.spectralCentroid(this.freqData);
    const zcr = this.zeroCrossingRate(this.timeData);

    // ----------------------------------------------------------------
    //  Variations & clarity
    // ----------------------------------------------------------------
    const pitchVar = this.variation(this.pitchHistory);
    const volVarVar = this.variation(this.volumeHistory);
    const clarity = this.clarityScore(snr, zcr, centroid, energy, confidence);

    return {
      pitch: Math.round(pitch),
      pitchVariation: Math.round(Math.min(100, pitchVar * 100)),
      volume: Math.round(volumeDB),
      volumeVariation: Math.round(Math.min(100, volVar * 100)),
      pace: 0, // external
      clarity: Math.round(Math.max(0, Math.min(100, clarity))),
      energy: Math.round(energy),
      spectralCentroid: Math.round(centroid),
      zeroCrossingRate: Math.round(zcr * 100),
      snr: Math.round(snr),
      confidence: Math.round(confidence * 100),
    };
  }

  cleanup(): void {
    try {
      this.audioContext?.close();
    } catch (_) {}
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.timeData = null;
    this.freqData = null;
    this.floatFreq = null;
    this.resetState();
  }

  // ------------------------------------------------------------------
  //  PRIVATE HELPERS
  // ------------------------------------------------------------------
  private defaultFeatures(): AudioFeatures {
    return {
      pitch: 0,
      pitchVariation: 0,
      volume: 0,
      volumeVariation: 0,
      pace: 0,
      clarity: 0,
      energy: 0,
      spectralCentroid: 0,
      zeroCrossingRate: 0,
      snr: 0,
      confidence: 0,
    };
  }

  private resetState(): void {
    this.pitchHistory = [];
    this.volumeHistory = [];
    this.energyHistory = [];
    this.noiseFloor = -60;
    this.noiseCalibrated = false;
    this.calibrationSamples = [];
    this.frameCount = 0;
  }

  // -------------------------- MPM Pitch --------------------------
  private mpmPitch(buffer: Float32Array): { pitch: number; confidence: number } {
    const sampleRate = this.audioContext!.sampleRate;
    const N = buffer.length;

    const minTau = Math.floor(sampleRate / this.MAX_PITCH);
    const maxTau = Math.floor(sampleRate / this.MIN_PITCH);
    const searchLen = Math.min(maxTau, Math.floor(N / 2));

    if (searchLen < minTau) return { pitch: 0, confidence: 0 };

    // ---- NSDF (Normalised Square Difference) ----
    const nsdf = new Float32Array(searchLen);
    let energy0 = 0;
    for (let i = 0; i < N; i++) energy0 += buffer[i] * buffer[i];

    for (let tau = 0; tau < searchLen; tau++) {
      let acf = 0,
        energy1 = 0;
      for (let i = 0; i < N - tau; i++) {
        acf += buffer[i] * buffer[i + tau];
        energy1 += buffer[i + tau] * buffer[i + tau];
      }
      const denom = Math.sqrt(energy0 * energy1);
      nsdf[tau] = denom > 0 ? (2 * acf) / denom : 0;
    }

    // ---- Find the strongest positive peak above cutoff ----
    let bestTau = 0;
    let bestVal = 0;
    for (let tau = minTau; tau < searchLen; tau++) {
      if (nsdf[tau] > bestVal && nsdf[tau] >= this.MPM_CUTOFF) {
        bestVal = nsdf[tau];
        bestTau = tau;
      }
    }
    if (bestVal === 0) return { pitch: 0, confidence: 0 };

    // ---- Parabolic interpolation ----
    if (bestTau > 0 && bestTau < searchLen - 1) {
      const a = nsdf[bestTau - 1];
      const b = nsdf[bestTau];
      const c = nsdf[bestTau + 1];
      const p = 0.5 * (a - c) / (a - 2 * b + c);
      bestTau += p;
    }

    const pitch = sampleRate / bestTau;

    // ---- HPS validation (quick check on the FFT) ----
    const hps = this.hpsValidate(pitch);

    const confidence = Math.min(1, (bestVal + hps) / 2);
    return { pitch: this.isValidPitch(pitch) ? pitch : 0, confidence };
  }

  /** Simple HPS on the *float* frequency data */
  private hpsValidate(pitch: number): number {
    if (!this.floatFreq || pitch === 0) return 0;
    const bin = Math.round((pitch * this.floatFreq.length) / this.audioContext!.sampleRate);
    if (bin >= this.floatFreq.length) return 0;

    let fund = this.floatFreq[bin] ?? -Infinity;
    if (!isFinite(fund)) fund = -Infinity;

    let harmSum = 0;
    let harmCnt = 0;
    for (let h = 2; h <= 5; h++) {
      const hi = bin * h;
      if (hi < this.floatFreq.length) {
        harmSum += this.floatFreq[hi];
        harmCnt++;
      }
    }
    const avgHarm = harmCnt ? harmSum / harmCnt : 0;
    return fund > -Infinity ? Math.min(1, avgHarm / (fund + 1e-9)) : 0;
  }

  // -------------------------- Basic features --------------------------
  private rms(buf: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  }

  private toDB(amp: number): number {
    return amp > 0 ? 20 * Math.log10(amp) : -Infinity;
  }

  private energy(freq: Uint8Array): number {
    let sum = 0;
    const limit = Math.min(512, freq.length);
    for (let i = 0; i < limit; i++) {
      const n = freq[i] / 255;
      sum += n * n;
    }
    return Math.sqrt(sum / limit) * 100;
  }

  private spectralCentroid(freq: Uint8Array): number {
    let num = 0,
      den = 0;
    for (let i = 0; i < freq.length; i++) {
      const m = freq[i];
      num += i * m;
      den += m;
    }
    return den ? num / den : 0;
  }

  private zeroCrossingRate(buf: Float32Array): number {
    let crosses = 0;
    for (let i = 1; i < buf.length; i++) {
      if ((buf[i] >= 0 && buf[i - 1] < 0) || (buf[i] < 0 && buf[i - 1] >= 0)) crosses++;
    }
    return crosses / buf.length;
  }

  private variation(hist: number[]): number {
    if (hist.length < 2) return 0;
    const mean = hist.reduce((a, b) => a + b, 0) / hist.length;
    if (mean === 0) return 0;
    const variance =
      hist.reduce((s, v) => s + (v - mean) * (v - mean), 0) / hist.length;
    return Math.sqrt(variance) / Math.abs(mean);
  }

  private clarityScore(
    snr: number,
    zcr: number,
    centroid: number,
    energy: number,
    conf: number
  ): number {
    const cSNR = Math.max(0, Math.min(100, ((snr + 5) / 25) * 100));
    const cZCR = Math.max(0, (1 - Math.min(zcr / 0.25, 1)) * 100);
    const cCent = Math.max(0, Math.min(100, (centroid / 150) * 100));
    const cEnergy = Math.max(0, Math.min(100, energy));
    const cConf = conf * 100;

    return (
      cSNR * 0.50 +
      cZCR * 0.15 +
      cCent * 0.10 +
      cEnergy * 0.10 +
      cConf * 0.15
    );
  }

  private calibrateNoise(db: number): void {
    if (!isFinite(db)) return;
    this.calibrationSamples.push(db);
    if (this.calibrationSamples.length >= 20) {
      const sorted = [...this.calibrationSamples].sort((a, b) => a - b);
      this.noiseFloor = sorted[Math.floor(sorted.length / 2)];
      this.noiseCalibrated = true;
      console.log(`Noise floor calibrated: ${this.noiseFloor.toFixed(1)} dB`);
    }
  }

  private isValidPitch(p: number): boolean {
    return isFinite(p) && p >= this.MIN_PITCH && p <= this.MAX_PITCH;
  }
}
