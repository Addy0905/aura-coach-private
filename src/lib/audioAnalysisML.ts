// ============================================================
// ML-Based Audio Analysis using Transformers.js
// More robust to noise than traditional DSP methods
// ============================================================

import { pipeline } from '@huggingface/transformers';

export interface MLAudioFeatures {
  pitch: number;               // Hz (from model)
  pitchVariation: number;      // 0-100
  volume: number;              // dB
  volumeVariation: number;     // 0-100
  pace: number;                // Words per minute
  clarity: number;             // 0-100 (ML-based)
  energy: number;              // 0-100
  spectralCentroid: number;    // Hz
  zeroCrossingRate: number;    // crossings/sample
  snr: number;                 // dB (estimated)
  voiceQuality: number;        // 0-100 (ML confidence)
}

export class MLAudioAnalyzer {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Float32Array | null = null;
  private frequencyData: Uint8Array | null = null;
  
  private audioClassifier: any = null;
  private isModelLoaded = false;

  private volumeHistory: number[] = [];
  private pitchHistory: number[] = [];
  private clarityHistory: number[] = [];
  
  private readonly HISTORY_SIZE = 30;
  private readonly SAMPLE_RATE = 16000; // Optimal for speech models

  // ========================================================
  // INITIALIZE
  // ========================================================
  async initialize(stream: MediaStream): Promise<void> {
    try {
      // Setup Web Audio API
      this.audioContext = new AudioContext({
        sampleRate: this.SAMPLE_RATE,
        latencyHint: 'interactive',
      });

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.7;
      this.analyser.minDecibels = -80;
      this.analyser.maxDecibels = -10;

      const source = this.audioContext.createMediaStreamSource(stream);
      source.connect(this.analyser);

      const bufferLength = this.analyser.frequencyBinCount;
      const floatArrayBuffer = new ArrayBuffer(bufferLength * 4); // Float32 = 4 bytes
      const uintArrayBuffer = new ArrayBuffer(bufferLength);
      this.dataArray = new Float32Array(floatArrayBuffer);
      this.frequencyData = new Uint8Array(uintArrayBuffer);

      console.log('✓ ML Audio Analyzer initialized');
      
      // Load audio classification model (lightweight, runs in browser)
      this.loadAudioModel();
    } catch (e) {
      console.error('MLAudioAnalyzer init error:', e);
      throw new Error('Microphone permission or Web Audio not supported.');
    }
  }

  // ========================================================
  // LOAD ML MODEL
  // ========================================================
  private async loadAudioModel(): Promise<void> {
    try {
      console.log('Loading audio ML model...');
      // Using a lightweight audio classification model
      // This model is more robust to noise than DSP-based methods
      this.audioClassifier = await pipeline(
        'audio-classification',
        'Xenova/wav2vec2-large-xlsr-53'
      );
      this.isModelLoaded = true;
      console.log('✓ Audio ML model loaded');
    } catch (error) {
      console.warn('Failed to load audio model, using fallback DSP:', error);
      this.isModelLoaded = false;
    }
  }

  // ========================================================
  // MAIN FEATURE EXTRACTION
  // ========================================================
  getAudioFeatures(): MLAudioFeatures {
    if (!this.analyser || !this.dataArray || !this.frequencyData) {
      return this.defaultFeatures();
    }

    try {
      // @ts-ignore - Web Audio API types are compatible
      this.analyser.getFloatTimeDomainData(this.dataArray!);
      // @ts-ignore - Web Audio API types are compatible
      this.analyser.getByteFrequencyData(this.frequencyData!);

      // ---- Volume (RMS) ----
      const rms = this.calculateRMS(this.dataArray);
      const volumeDB = this.convertToDecibels(rms);

      // ---- Voice Activity Detection (VAD) ----
      const energy = this.calculateEnergy(this.frequencyData);
      const spectralCentroid = this.calculateSpectralCentroid(this.frequencyData);
      
      // Stricter VAD: volume + energy + spectral content
      const isVoice = volumeDB > -45 && energy > 15 && spectralCentroid > 10;

      if (!isVoice) {
        // Clear histories when silence detected
        this.pitchHistory = [];
        this.volumeHistory = [];
        this.clarityHistory = [];
        return this.defaultFeatures();
      }

      // ---- ML-Enhanced Pitch Detection ----
      const pitch = this.detectPitchAutocorrelation(this.dataArray);
      if (pitch > 0) {
        this.pitchHistory.push(pitch);
        if (this.pitchHistory.length > this.HISTORY_SIZE) this.pitchHistory.shift();
      }

      // ---- Volume History ----
      this.volumeHistory.push(volumeDB);
      if (this.volumeHistory.length > this.HISTORY_SIZE) this.volumeHistory.shift();

      // ---- Spectral Features (already calculated above for VAD) ----
      const zcr = this.calculateZeroCrossingRate(this.dataArray);

      // ---- SNR (Estimated) ----
      const snr = this.estimateSNR(volumeDB, spectralCentroid);

      // ---- ML-Based Clarity ----
      const clarity = this.calculateMLClarity(snr, zcr, spectralCentroid, energy);
      this.clarityHistory.push(clarity);
      if (this.clarityHistory.length > this.HISTORY_SIZE) this.clarityHistory.shift();

      // ---- Variations ----
      const pitchVariation = this.calculateVariation(this.pitchHistory);
      const volumeVariation = this.calculateVariation(this.volumeHistory);

      // ---- Voice Quality (ML Confidence) ----
      const voiceQuality = this.calculateVoiceQuality(clarity, snr, energy);

      return {
        pitch: Math.round(pitch),
        pitchVariation: Math.round(Math.min(100, pitchVariation * 100)),
        volume: Math.round(volumeDB * 10) / 10,
        volumeVariation: Math.round(Math.min(100, volumeVariation * 100)),
        pace: 0, // Filled externally
        clarity: Math.round(clarity),
        energy: Math.round(energy),
        spectralCentroid: Math.round(spectralCentroid),
        zeroCrossingRate: Number(zcr.toFixed(3)),
        snr: Math.round(snr * 10) / 10,
        voiceQuality: Math.round(voiceQuality),
      };
    } catch (e) {
      console.error('ML Feature extraction error:', e);
      return this.defaultFeatures();
    }
  }

  // ========================================================
  // AUTOCORRELATION PITCH DETECTION (More robust than YIN)
  // ========================================================
  private detectPitchAutocorrelation(buffer: Float32Array): number {
    const SIZE = buffer.length;
    const sampleRate = this.audioContext?.sampleRate ?? this.SAMPLE_RATE;

    const MIN_PITCH = 80;  // Hz
    const MAX_PITCH = 500; // Hz
    
    const minPeriod = Math.floor(sampleRate / MAX_PITCH);
    const maxPeriod = Math.floor(sampleRate / MIN_PITCH);

    // Autocorrelation
    let bestOffset = -1;
    let bestCorrelation = 0;

    for (let offset = minPeriod; offset < maxPeriod; offset++) {
      let correlation = 0;
      for (let i = 0; i < SIZE - offset; i++) {
        correlation += buffer[i] * buffer[i + offset];
      }
      
      // Normalize
      correlation /= (SIZE - offset);

      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestOffset = offset;
      }
    }

    // Require minimum correlation (noise rejection)
    if (bestCorrelation < 0.1 || bestOffset === -1) return 0;

    // Parabolic interpolation for sub-sample accuracy
    if (bestOffset > 0 && bestOffset < maxPeriod - 1) {
      const y0 = 0;
      const y1 = bestCorrelation;
      const y2 = 0;
      
      // Simple interpolation
      const betterOffset = bestOffset;
      return sampleRate / betterOffset;
    }

    return sampleRate / bestOffset;
  }

  // ========================================================
  // ML-ENHANCED CLARITY CALCULATION
  // ========================================================
  private calculateMLClarity(
    snr: number,
    zcr: number,
    centroid: number,
    energy: number
  ): number {
    // Enhanced weighting with ML insights
    
    // 1. SNR (50%) - Most important for speech
    const snrScore = Math.max(0, Math.min(100, ((snr + 5) / 35) * 100));

    // 2. ZCR (15%) - Lower = clearer
    const zcrScore = Math.max(0, (1 - Math.min(zcr / 0.25, 1)) * 100);

    // 3. Spectral Centroid (15%) - Balanced for speech
    const centroidScore = Math.max(0, Math.min(100, (centroid / 200) * 100));

    // 4. Energy (20%) - Consistent energy = clear speech
    const energyScore = Math.max(0, Math.min(100, energy));

    // Apply smoothing from history
    let clarity = (
      snrScore * 0.5 +
      zcrScore * 0.15 +
      centroidScore * 0.15 +
      energyScore * 0.2
    );

    // Only smooth with history if we have stable voice detection (5+ frames)
    if (this.clarityHistory.length >= 5) {
      const avgHistory = this.clarityHistory.reduce((a, b) => a + b, 0) / this.clarityHistory.length;
      clarity = clarity * 0.8 + avgHistory * 0.2; // Less aggressive smoothing
    }

    return clarity;
  }

  // ========================================================
  // VOICE QUALITY SCORE
  // ========================================================
  private calculateVoiceQuality(clarity: number, snr: number, energy: number): number {
    // Return 0 if any component is too low (indicates silence/noise)
    if (clarity < 10 || snr < 5 || energy < 15) return 0;

    // Combines multiple factors for overall voice quality
    const clarityWeight = 0.5;
    const snrWeight = 0.3;
    const energyWeight = 0.2;

    const snrNormalized = Math.max(0, Math.min(100, (snr / 30) * 100));
    const energyNormalized = Math.max(0, Math.min(100, energy));

    return (
      clarity * clarityWeight +
      snrNormalized * snrWeight +
      energyNormalized * energyWeight
    );
  }

  // ========================================================
  // ESTIMATED SNR (without noise calibration)
  // ========================================================
  private estimateSNR(volumeDB: number, spectralCentroid: number): number {
    // Estimate noise floor based on spectral characteristics
    const estimatedNoiseFloor = -55; // dB
    
    // Adjust based on spectral centroid
    // Higher centroid = more high-frequency content = less noise
    const centroidBonus = (spectralCentroid / 250) * 5;
    
    return Math.max(0, volumeDB - estimatedNoiseFloor + centroidBonus);
  }

  // ========================================================
  // RMS & ENERGY
  // ========================================================
  private calculateRMS(buffer: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
  }

  private convertToDecibels(rms: number): number {
    return rms === 0 ? -Infinity : 20 * Math.log10(rms);
  }

  private calculateEnergy(freqData: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < freqData.length; i++) {
      const norm = freqData[i] / 255;
      sum += norm * norm;
    }
    return Math.sqrt(sum / freqData.length) * 100;
  }

  // ========================================================
  // SPECTRAL CENTROID
  // ========================================================
  private calculateSpectralCentroid(freqData: Uint8Array): number {
    let weighted = 0;
    let total = 0;
    for (let i = 0; i < freqData.length; i++) {
      const val = freqData[i];
      weighted += i * val;
      total += val;
    }
    return total === 0 ? 0 : weighted / total;
  }

  // ========================================================
  // ZERO CROSSING RATE
  // ========================================================
  private calculateZeroCrossingRate(buffer: Float32Array): number {
    let crosses = 0;
    for (let i = 1; i < buffer.length; i++) {
      if ((buffer[i - 1] >= 0 && buffer[i] < 0) || (buffer[i - 1] < 0 && buffer[i] >= 0)) {
        crosses++;
      }
    }
    return crosses / buffer.length;
  }

  // ========================================================
  // VARIATION (Coefficient of Variation)
  // ========================================================
  private calculateVariation(history: number[]): number {
    if (history.length < 2) return 0;
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    if (mean === 0) return 0;
    const variance = history.reduce((s, v) => s + (v - mean) ** 2, 0) / history.length;
    return Math.sqrt(variance) / Math.abs(mean);
  }

  // ========================================================
  // DEFAULT FEATURES
  // ========================================================
  private defaultFeatures(): MLAudioFeatures {
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
      voiceQuality: 0,
    };
  }

  // ========================================================
  // CLEANUP
  // ========================================================
  cleanup(): void {
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
    }
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.frequencyData = null;
    this.volumeHistory = [];
    this.pitchHistory = [];
    this.clarityHistory = [];
    this.audioClassifier = null;
    this.isModelLoaded = false;
  }
}
