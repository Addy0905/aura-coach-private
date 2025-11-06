// Advanced audio analysis using Web Audio API (browser-based Librosa equivalent)

export interface AudioFeatures {
  pitch: number; // Average pitch in Hz (YIN Algorithm)
  pitchVariation: number; // Variation in pitch (0-100)
  volume: number; // RMS amplitude in dB
  volumeVariation: number; // Variation in volume
  pace: number; // Words per minute
  clarity: number; // SNR-based clarity score (0-100)
  energy: number; // Audio energy level
  spectralCentroid: number; // Brightness of sound
  zeroCrossingRate: number; // Rate of sign changes
  snr: number; // Signal-to-Noise Ratio in dB
}

export class AudioAnalyzer {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private frequencyData: Uint8Array | null = null;
  private pitchHistory: number[] = [];
  private volumeHistory: number[] = [];
  private energyHistory: number[] = [];
  private noiseFloor: number = 0;
  private noiseCalibrated = false;

  initialize(stream: MediaStream) {
    this.audioContext = new AudioContext();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    
    const source = this.audioContext.createMediaStreamSource(stream);
    source.connect(this.analyser);
    
    this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
  }

  getAudioFeatures(): AudioFeatures {
    if (!this.analyser || !this.dataArray || !this.frequencyData) {
      return this.getDefaultFeatures();
    }

    // Get time domain data
    this.analyser.getByteTimeDomainData(this.dataArray as any);
    // Get frequency domain data
    this.analyser.getByteFrequencyData(this.frequencyData as any);

    // Calculate pitch using YIN-like autocorrelation algorithm
    const pitch = this.detectPitchYIN(this.dataArray);
    this.pitchHistory.push(pitch);
    if (this.pitchHistory.length > 50) this.pitchHistory.shift();

    // Calculate volume using RMS (Root Mean Square) amplitude
    const volume = this.calculateRMS(this.dataArray);
    const volumeDB = 20 * Math.log10(volume / 255);
    this.volumeHistory.push(volumeDB);
    if (this.volumeHistory.length > 50) this.volumeHistory.shift();

    // Calibrate noise floor (first 10 samples)
    if (!this.noiseCalibrated && this.volumeHistory.length >= 10) {
      const sortedVolumes = [...this.volumeHistory].sort((a, b) => a - b);
      this.noiseFloor = sortedVolumes[2]; // 3rd quietest sample
      this.noiseCalibrated = true;
    }

    // Calculate SNR (Signal-to-Noise Ratio)
    const snr = this.calculateSNR(volumeDB);

    // Calculate energy
    const energy = this.calculateEnergy(this.frequencyData);
    this.energyHistory.push(energy);
    if (this.energyHistory.length > 30) this.energyHistory.shift();

    // Calculate spectral centroid (brightness)
    const spectralCentroid = this.calculateSpectralCentroid(this.frequencyData);

    // Calculate zero crossing rate
    const zcr = this.calculateZeroCrossingRate(this.dataArray);

    // Calculate variations
    const pitchVariation = this.calculateVariation(this.pitchHistory);
    const volumeVariation = this.calculateVariation(this.volumeHistory);

    // Calculate clarity from SNR, ZCR, and spectral features
    // Only calculate clarity if we have calibrated noise and signal is strong
    let clarity = 0;
    if (this.noiseCalibrated && volumeDB > -50) {
      const clarityFromSNR = Math.max(0, Math.min(100, ((snr + 10) / 30) * 100)); // SNR -10 to 20 dB maps to 0-100
      const clarityFromZCR = Math.max(0, (1 - Math.min(zcr / 0.3, 1)) * 100);
      const clarityFromSpectral = Math.max(0, Math.min(100, (spectralCentroid / 200) * 100));
      clarity = (clarityFromSNR * 0.6 + clarityFromZCR * 0.2 + clarityFromSpectral * 0.2);
    }

    return {
      pitch: Math.round(pitch),
      pitchVariation: Math.round(pitchVariation * 100),
      volume: Math.round(volumeDB),
      volumeVariation: Math.round(volumeVariation * 100),
      pace: 0, // Will be calculated from transcript with syllable estimation
      clarity: Math.round(clarity),
      energy: Math.round(energy),
      spectralCentroid: Math.round(spectralCentroid),
      zeroCrossingRate: Math.round(zcr * 1000) / 1000,
      snr: Math.round(snr * 10) / 10,
    };
  }

  // YIN Algorithm for pitch detection (more accurate than basic autocorrelation)
  private detectPitchYIN(buffer: Uint8Array): number {
    const SIZE = buffer.length;
    const MAX_SAMPLES = Math.floor(SIZE / 2);
    const threshold = 0.1;
    
    // Step 1: Calculate difference function
    const difference = new Float32Array(MAX_SAMPLES);
    for (let tau = 0; tau < MAX_SAMPLES; tau++) {
      let sum = 0;
      for (let i = 0; i < MAX_SAMPLES; i++) {
        const delta = ((buffer[i] - 128) / 128) - ((buffer[i + tau] - 128) / 128);
        sum += delta * delta;
      }
      difference[tau] = sum;
    }
    
    // Step 2: Cumulative mean normalized difference
    const cmndf = new Float32Array(MAX_SAMPLES);
    cmndf[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < MAX_SAMPLES; tau++) {
      runningSum += difference[tau];
      cmndf[tau] = difference[tau] / (runningSum / tau);
    }
    
    // Step 3: Absolute threshold
    let tau = 2; // Start from 2 to avoid zero
    while (tau < MAX_SAMPLES) {
      if (cmndf[tau] < threshold) {
        while (tau + 1 < MAX_SAMPLES && cmndf[tau + 1] < cmndf[tau]) {
          tau++;
        }
        const sampleRate = this.audioContext?.sampleRate || 44100;
        return sampleRate / tau;
      }
      tau++;
    }
    
    return 0; // No pitch detected
  }

  // Calculate Signal-to-Noise Ratio
  private calculateSNR(signalDB: number): number {
    if (!this.noiseCalibrated) return 0;
    return signalDB - this.noiseFloor;
  }

  private calculateRMS(buffer: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      const normalized = (buffer[i] - 128) / 128;
      sum += normalized * normalized;
    }
    return Math.sqrt(sum / buffer.length) * 255;
  }

  private calculateEnergy(frequencyData: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < frequencyData.length; i++) {
      sum += frequencyData[i] * frequencyData[i];
    }
    return Math.sqrt(sum / frequencyData.length);
  }

  private calculateSpectralCentroid(frequencyData: Uint8Array): number {
    let weightedSum = 0;
    let sum = 0;
    
    for (let i = 0; i < frequencyData.length; i++) {
      weightedSum += i * frequencyData[i];
      sum += frequencyData[i];
    }
    
    return sum === 0 ? 0 : weightedSum / sum;
  }

  private calculateZeroCrossingRate(buffer: Uint8Array): number {
    let crossings = 0;
    for (let i = 1; i < buffer.length; i++) {
      if ((buffer[i] >= 128 && buffer[i - 1] < 128) || 
          (buffer[i] < 128 && buffer[i - 1] >= 128)) {
        crossings++;
      }
    }
    return crossings / buffer.length;
  }

  private calculateVariation(history: number[]): number {
    if (history.length < 2) return 0;
    
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / history.length;
    const stdDev = Math.sqrt(variance);
    
    return mean === 0 ? 0 : stdDev / Math.abs(mean);
  }

  private getDefaultFeatures(): AudioFeatures {
    return {
      pitch: 0,
      pitchVariation: 0,
      volume: -60,
      volumeVariation: 0,
      pace: 0,
      clarity: 0,
      energy: 0,
      spectralCentroid: 0,
      zeroCrossingRate: 0,
      snr: 0,
    };
  }

  cleanup() {
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.analyser = null;
    this.dataArray = null;
    this.frequencyData = null;
    this.pitchHistory = [];
    this.volumeHistory = [];
    this.energyHistory = [];
    this.noiseFloor = 0;
    this.noiseCalibrated = false;
  }
}
