// Advanced audio analysis using Web Audio API (browser-based Librosa equivalent)
// Algorithms: YIN pitch detection, RMS amplitude, SNR, Spectral analysis, ZCR

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
  private dataArray: Uint8Array<ArrayBuffer> | null = null;
  private frequencyData: Uint8Array<ArrayBuffer> | null = null;
  private pitchHistory: number[] = [];
  private volumeHistory: number[] = [];
  private energyHistory: number[] = [];
  private noiseFloor: number = -60;
  private noiseCalibrated = false;
  private calibrationSamples: number[] = [];
  private readonly VOICE_THRESHOLD_DB = -45; // Minimum dB to consider as voice
  private readonly MIN_PITCH = 80; // Hz - below this is likely noise
  private readonly MAX_PITCH = 500; // Hz - above this is likely noise for speech

  /**
   * Initialize the audio analyzer with a media stream
   * Creates AudioContext, AnalyserNode, and connects the audio pipeline
   */
  initialize(stream: MediaStream): void {
    try {
      // Create audio context with optimal sample rate
      this.audioContext = new AudioContext({ 
        sampleRate: 48000,
        latencyHint: 'interactive'
      });
      
      // Create analyser node with optimal FFT size for speech analysis
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048; // Good balance for pitch detection (80-500 Hz)
      this.analyser.smoothingTimeConstant = 0.8; // Smooth out noise
      this.analyser.minDecibels = -90;
      this.analyser.maxDecibels = -10;
      
      // Connect audio source to analyser
      const source = this.audioContext.createMediaStreamSource(stream);
      source.connect(this.analyser);
      
      // Initialize data arrays
      const bufferLength = this.analyser.frequencyBinCount;
      const buffer = new ArrayBuffer(bufferLength);
      this.dataArray = new Uint8Array(buffer);
      const buffer2 = new ArrayBuffer(bufferLength);
      this.frequencyData = new Uint8Array(buffer2);
      
      // Reset calibration
      this.noiseCalibrated = false;
      this.calibrationSamples = [];
    } catch (error) {
      console.error('Failed to initialize audio analyzer:', error);
      throw new Error('Audio initialization failed. Please check microphone permissions.');
    }
  }

  /**
   * Get current audio features
   * Returns default values (zeros) when no voice is detected
   */
  getAudioFeatures(): AudioFeatures {
    if (!this.analyser || !this.dataArray || !this.frequencyData) {
      return this.getDefaultFeatures();
    }

    try {
      // Get time domain data (waveform)
      this.analyser.getByteTimeDomainData(this.dataArray);
      
      // Get frequency domain data (spectrum)
      this.analyser.getByteFrequencyData(this.frequencyData);

      // Calculate volume first to determine if voice is present
      const volume = this.calculateRMS(this.dataArray);
      const volumeDB = this.convertToDecibels(volume);
      
      // Calibrate noise floor during initial samples
      if (!this.noiseCalibrated) {
        this.calibrateNoiseFloor(volumeDB);
      }

      // Calculate SNR to determine if voice is present
      const snr = this.calculateSNR(volumeDB);
      
      // Check if voice is detected (above threshold and reasonable SNR)
      const isVoiceDetected = volumeDB > this.VOICE_THRESHOLD_DB && snr > 0;

      // If no voice detected, return zeros
      if (!isVoiceDetected) {
        return this.getDefaultFeatures();
      }

      // Voice detected - calculate all features
      
      // 1. Pitch Detection using YIN Algorithm
      const pitch = this.detectPitchYIN(this.dataArray);
      const validPitch = this.isValidPitch(pitch) ? pitch : 0;
      
      if (validPitch > 0) {
        this.pitchHistory.push(validPitch);
        if (this.pitchHistory.length > 50) this.pitchHistory.shift();
      }

      // 2. Volume tracking
      this.volumeHistory.push(volumeDB);
      if (this.volumeHistory.length > 50) this.volumeHistory.shift();

      // 3. Energy calculation
      const energy = this.calculateEnergy(this.frequencyData);
      this.energyHistory.push(energy);
      if (this.energyHistory.length > 30) this.energyHistory.shift();

      // 4. Spectral Centroid (brightness of sound)
      const spectralCentroid = this.calculateSpectralCentroid(this.frequencyData);

      // 5. Zero Crossing Rate (voice activity indicator)
      const zcr = this.calculateZeroCrossingRate(this.dataArray);

      // 6. Calculate variations
      const pitchVariation = this.calculateVariation(this.pitchHistory);
      const volumeVariation = this.calculateVariation(this.volumeHistory);

      // 7. Calculate clarity score (multi-factor quality metric)
      const clarity = this.calculateClarity(snr, zcr, spectralCentroid, energy);

      return {
        pitch: Math.round(validPitch),
        pitchVariation: Math.round(Math.min(100, pitchVariation * 100)),
        volume: Math.round(volumeDB),
        volumeVariation: Math.round(Math.min(100, volumeVariation * 100)),
        pace: 0, // Calculated externally from transcript
        clarity: Math.round(Math.max(0, Math.min(100, clarity))),
        energy: Math.round(energy),
        spectralCentroid: Math.round(spectralCentroid),
        zeroCrossingRate: Math.round(zcr * 100),
        snr: Math.round(snr),
      };
    } catch (error) {
      console.error('Error analyzing audio features:', error);
      return this.getDefaultFeatures();
    }
  }

  /**
   * YIN Algorithm for pitch detection
   * More accurate than basic autocorrelation, especially for low pitches
   * Steps: 1) Difference function, 2) Cumulative mean normalized difference, 3) Absolute threshold
   */
  private detectPitchYIN(buffer: Uint8Array): number {
    const SIZE = buffer.length;
    const sampleRate = this.audioContext?.sampleRate || 48000;
    
    // Calculate search range based on expected pitch (80-500 Hz)
    const minPeriod = Math.floor(sampleRate / this.MAX_PITCH);
    const maxPeriod = Math.floor(sampleRate / this.MIN_PITCH);
    const searchSize = Math.min(maxPeriod, Math.floor(SIZE / 2));
    
    if (searchSize < minPeriod) return 0;

    const threshold = 0.15; // YIN threshold for pitch detection
    
    // Step 1: Calculate difference function (squared difference)
    const difference = new Float32Array(searchSize);
    for (let tau = 0; tau < searchSize; tau++) {
      let sum = 0;
      for (let i = 0; i < searchSize; i++) {
        const delta = ((buffer[i] - 128) / 128) - ((buffer[i + tau] - 128) / 128);
        sum += delta * delta;
      }
      difference[tau] = sum;
    }
    
    // Step 2: Cumulative mean normalized difference function (CMNDF)
    const cmndf = new Float32Array(searchSize);
    cmndf[0] = 1;
    let runningSum = 0;
    
    for (let tau = 1; tau < searchSize; tau++) {
      runningSum += difference[tau];
      cmndf[tau] = difference[tau] / (runningSum / tau);
    }
    
    // Step 3: Absolute threshold - find first minimum below threshold
    let tau = minPeriod;
    while (tau < searchSize) {
      if (cmndf[tau] < threshold) {
        // Find local minimum (parabolic interpolation for sub-sample accuracy)
        while (tau + 1 < searchSize && cmndf[tau + 1] < cmndf[tau]) {
          tau++;
        }
        
        // Parabolic interpolation for better accuracy
        if (tau > 0 && tau < searchSize - 1) {
          const betterTau = tau + (cmndf[tau + 1] - cmndf[tau - 1]) / (2 * (2 * cmndf[tau] - cmndf[tau - 1] - cmndf[tau + 1]));
          const pitch = sampleRate / betterTau;
          return pitch;
        }
        
        return sampleRate / tau;
      }
      tau++;
    }
    
    return 0; // No pitch detected
  }

  /**
   * Calculate Signal-to-Noise Ratio
   * Measures how much signal stands out from background noise
   */
  private calculateSNR(signalDB: number): number {
    if (!this.noiseCalibrated) return 0;
    return signalDB - this.noiseFloor;
  }

  /**
   * Calculate RMS (Root Mean Square) amplitude
   * Standard measure of audio signal strength
   */
  private calculateRMS(buffer: Uint8Array): number {
    let sum = 0;
    const length = buffer.length;
    
    for (let i = 0; i < length; i++) {
      const normalized = (buffer[i] - 128) / 128;
      sum += normalized * normalized;
    }
    
    return Math.sqrt(sum / length);
  }

  /**
   * Convert RMS amplitude to decibels
   * dB = 20 * log10(amplitude)
   */
  private convertToDecibels(amplitude: number): number {
    if (amplitude === 0) return -Infinity;
    return 20 * Math.log10(amplitude);
  }

  /**
   * Calculate audio energy from frequency spectrum
   * Measures overall signal intensity
   */
  private calculateEnergy(frequencyData: Uint8Array): number {
    let sum = 0;
    const length = frequencyData.length;
    
    for (let i = 0; i < length; i++) {
      const normalized = frequencyData[i] / 255;
      sum += normalized * normalized;
    }
    
    return Math.sqrt(sum / length) * 100;
  }

  /**
   * Calculate Spectral Centroid
   * Measures the "brightness" of the sound (center of mass of spectrum)
   * Higher values = brighter/higher frequency content
   */
  private calculateSpectralCentroid(frequencyData: Uint8Array): number {
    let weightedSum = 0;
    let sum = 0;
    const length = frequencyData.length;
    
    for (let i = 0; i < length; i++) {
      weightedSum += i * frequencyData[i];
      sum += frequencyData[i];
    }
    
    return sum === 0 ? 0 : weightedSum / sum;
  }

  /**
   * Calculate Zero Crossing Rate
   * Measures how often the signal changes sign (crosses zero)
   * Higher ZCR = more noisy/high-frequency content
   */
  private calculateZeroCrossingRate(buffer: Uint8Array): number {
    let crossings = 0;
    const length = buffer.length;
    
    for (let i = 1; i < length; i++) {
      if ((buffer[i] >= 128 && buffer[i - 1] < 128) || 
          (buffer[i] < 128 && buffer[i - 1] >= 128)) {
        crossings++;
      }
    }
    
    return crossings / length;
  }

  /**
   * Calculate coefficient of variation
   * Normalized measure of dispersion (standard deviation / mean)
   */
  private calculateVariation(history: number[]): number {
    if (history.length < 2) return 0;
    
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    if (mean === 0) return 0;
    
    const variance = history.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / history.length;
    const stdDev = Math.sqrt(variance);
    
    return stdDev / Math.abs(mean);
  }

  /**
   * Calculate multi-factor clarity score
   * Combines SNR, ZCR, spectral centroid, and energy
   */
  private calculateClarity(snr: number, zcr: number, spectralCentroid: number, energy: number): number {
    // SNR contribution (60% weight) - good speech has SNR 10-30 dB
    const clarityFromSNR = Math.max(0, Math.min(100, ((snr + 10) / 30) * 100));
    
    // ZCR contribution (20% weight) - lower is better for speech
    const clarityFromZCR = Math.max(0, (1 - Math.min(zcr / 0.3, 1)) * 100);
    
    // Spectral centroid contribution (10% weight)
    const clarityFromSpectral = Math.max(0, Math.min(100, (spectralCentroid / 200) * 100));
    
    // Energy contribution (10% weight)
    const clarityFromEnergy = Math.max(0, Math.min(100, energy));
    
    return (clarityFromSNR * 0.6 + clarityFromZCR * 0.2 + clarityFromSpectral * 0.1 + clarityFromEnergy * 0.1);
  }

  /**
   * Calibrate noise floor from initial quiet samples
   * Uses median of first 20 samples to avoid outliers
   */
  private calibrateNoiseFloor(volumeDB: number): void {
    this.calibrationSamples.push(volumeDB);
    
    if (this.calibrationSamples.length >= 20) {
      // Use median instead of mean to be robust against outliers
      const sorted = [...this.calibrationSamples].sort((a, b) => a - b);
      this.noiseFloor = sorted[Math.floor(sorted.length / 2)];
      this.noiseCalibrated = true;
    }
  }

  /**
   * Validate pitch is in expected human speech range
   */
  private isValidPitch(pitch: number): boolean {
    return pitch >= this.MIN_PITCH && pitch <= this.MAX_PITCH;
  }

  /**
   * Return default features (all zeros) when no voice detected
   */
  private getDefaultFeatures(): AudioFeatures {
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
    };
  }

  /**
   * Clean up audio resources
   * Call this when stopping audio analysis
   */
  cleanup(): void {
    try {
      if (this.audioContext && this.audioContext.state !== 'closed') {
        this.audioContext.close();
      }
    } catch (error) {
      console.error('Error closing audio context:', error);
    }
    
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.frequencyData = null;
    this.pitchHistory = [];
    this.volumeHistory = [];
    this.energyHistory = [];
    this.noiseFloor = -60;
    this.noiseCalibrated = false;
    this.calibrationSamples = [];
  }
}
