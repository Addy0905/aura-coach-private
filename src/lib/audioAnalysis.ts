// Advanced audio analysis using Web Audio API (browser-based Librosa equivalent)

export interface AudioFeatures {
  pitch: number; // Average pitch in Hz
  pitchVariation: number; // Variation in pitch (0-100)
  volume: number; // Average volume in dB
  volumeVariation: number; // Variation in volume
  pace: number; // Words per minute
  clarity: number; // Pronunciation clarity score (0-100)
  energy: number; // Audio energy level
  spectralCentroid: number; // Brightness of sound
  zeroCrossingRate: number; // Rate of sign changes
}

export class AudioAnalyzer {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private frequencyData: Uint8Array | null = null;
  private pitchHistory: number[] = [];
  private volumeHistory: number[] = [];
  private energyHistory: number[] = [];

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

    // Calculate pitch using autocorrelation
    const pitch = this.detectPitch(this.dataArray);
    this.pitchHistory.push(pitch);
    if (this.pitchHistory.length > 50) this.pitchHistory.shift();

    // Calculate volume (RMS)
    const volume = this.calculateRMS(this.dataArray);
    const volumeDB = 20 * Math.log10(volume / 255);
    this.volumeHistory.push(volumeDB);
    if (this.volumeHistory.length > 50) this.volumeHistory.shift();

    // Calculate energy
    const energy = this.calculateEnergy(this.frequencyData);
    this.energyHistory.push(energy);
    if (this.energyHistory.length > 30) this.energyHistory.shift();

    // Calculate spectral centroid (brightness)
    const spectralCentroid = this.calculateSpectralCentroid(this.frequencyData);

    // Calculate zero crossing rate (clarity indicator)
    const zcr = this.calculateZeroCrossingRate(this.dataArray);

    // Calculate variations
    const pitchVariation = this.calculateVariation(this.pitchHistory);
    const volumeVariation = this.calculateVariation(this.volumeHistory);

    // Estimate clarity from ZCR and spectral features
    const clarity = Math.min(100, (1 - zcr / 0.5) * 50 + (spectralCentroid / 255) * 50);

    return {
      pitch: Math.round(pitch),
      pitchVariation: Math.round(pitchVariation * 100),
      volume: Math.round(volumeDB),
      volumeVariation: Math.round(volumeVariation * 100),
      pace: 0, // Will be calculated from transcript
      clarity: Math.max(25, Math.round(clarity)),
      energy: Math.round(energy),
      spectralCentroid: Math.round(spectralCentroid),
      zeroCrossingRate: Math.round(zcr * 1000) / 1000,
    };
  }

  private detectPitch(buffer: Uint8Array): number {
    // Autocorrelation method for pitch detection
    const SIZE = buffer.length;
    const MAX_SAMPLES = Math.floor(SIZE / 2);
    let best_offset = -1;
    let best_correlation = 0;
    let rms = 0;
    
    for (let i = 0; i < SIZE; i++) {
      const val = (buffer[i] - 128) / 128;
      rms += val * val;
    }
    rms = Math.sqrt(rms / SIZE);
    
    if (rms < 0.01) return 0; // Not enough signal
    
    let lastCorrelation = 1;
    for (let offset = 1; offset < MAX_SAMPLES; offset++) {
      let correlation = 0;
      for (let i = 0; i < MAX_SAMPLES; i++) {
        correlation += Math.abs(((buffer[i] - 128) / 128) - ((buffer[i + offset] - 128) / 128));
      }
      correlation = 1 - (correlation / MAX_SAMPLES);
      
      if (correlation > 0.9 && correlation > lastCorrelation) {
        const foundGoodCorrelation = correlation > best_correlation;
        if (foundGoodCorrelation) {
          best_correlation = correlation;
          best_offset = offset;
        }
      }
      lastCorrelation = correlation;
    }
    
    if (best_offset === -1) return 0;
    
    const sampleRate = this.audioContext?.sampleRate || 44100;
    return sampleRate / best_offset;
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
      clarity: 25,
      energy: 0,
      spectralCentroid: 0,
      zeroCrossingRate: 0,
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
  }
}
