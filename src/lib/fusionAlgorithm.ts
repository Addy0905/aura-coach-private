// Multi-Modal Fusion Algorithm for Persona AI
// Implements weighted fusion, temporal smoothing, and confidence scoring

export interface RawMetrics {
  // Vision metrics from MediaPipe
  eyeContact: number; // 0-100
  emotion: string;
  emotionConfidence: number; // 0-1
  postureScore: number; // 0-100
  shoulderAlignment: number; // 0-100
  headPosition: number; // 0-100
  gestureVariety: number; // 0-100
  handVisibility: number; // 0-100
  
  // Audio metrics from Web Audio API
  pitch: number; // Hz
  pitchVariation: number; // 0-100
  volume: number; // dB
  volumeVariation: number; // 0-100
  clarity: number; // 0-100 (from audio analysis)
  energy: number; // Audio energy
  
  // Speech metrics from Web Speech API
  wordsPerMinute: number;
  fillerCount: number;
  fillerPercentage: number;
  clarityScore: number; // 0-100 (from speech)
  fluencyScore: number; // 0-100
  articulationScore: number; // 0-100
}

export interface FusedMetrics {
  eyeContact: number;
  posture: number;
  bodyLanguage: number;
  facialExpression: number;
  voiceQuality: number;
  speechClarity: number;
  contentEngagement: number;
  overallScore: number;
  confidence: number;
}

export interface ContextWeights {
  professional: {
    eyeContact: number;
    posture: number;
    bodyLanguage: number;
    facialExpression: number;
    voiceQuality: number;
    speechClarity: number;
    contentEngagement: number;
  };
  presentation: {
    eyeContact: number;
    posture: number;
    bodyLanguage: number;
    facialExpression: number;
    voiceQuality: number;
    speechClarity: number;
    contentEngagement: number;
  };
  casual: {
    eyeContact: number;
    posture: number;
    bodyLanguage: number;
    facialExpression: number;
    voiceQuality: number;
    speechClarity: number;
    contentEngagement: number;
  };
}

// Weight configurations for different contexts
const CONTEXT_WEIGHTS: ContextWeights = {
  professional: {
    eyeContact: 0.20,
    posture: 0.15,
    bodyLanguage: 0.10,
    facialExpression: 0.15,
    voiceQuality: 0.15,
    speechClarity: 0.15,
    contentEngagement: 0.10,
  },
  presentation: {
    eyeContact: 0.25,
    posture: 0.15,
    bodyLanguage: 0.15,
    facialExpression: 0.10,
    voiceQuality: 0.10,
    speechClarity: 0.15,
    contentEngagement: 0.10,
  },
  casual: {
    eyeContact: 0.15,
    posture: 0.10,
    bodyLanguage: 0.10,
    facialExpression: 0.20,
    voiceQuality: 0.15,
    speechClarity: 0.15,
    contentEngagement: 0.15,
  },
};

export class FusionAlgorithm {
  private context: keyof ContextWeights = 'presentation';
  private history: FusedMetrics[] = [];
  private readonly HISTORY_SIZE = 10; // For temporal smoothing (last 10 frames)
  private readonly MIN_SCORE = 25; // Minimum threshold

  setContext(context: keyof ContextWeights) {
    this.context = context;
  }

  // Step 1: Normalize all metrics to 0-100 scale
  private normalizeMetrics(raw: RawMetrics): Record<string, number> {
    return {
      // Vision metrics (already 0-100)
      eyeContact: this.clamp(raw.eyeContact, 0, 100),
      postureScore: this.clamp(raw.postureScore, 0, 100),
      shoulderAlignment: this.clamp(raw.shoulderAlignment, 0, 100),
      headPosition: this.clamp(raw.headPosition, 0, 100),
      gestureVariety: this.clamp(raw.gestureVariety, 0, 100),
      handVisibility: this.clamp(raw.handVisibility, 0, 100),
      emotionConfidence: raw.emotionConfidence * 100, // Convert 0-1 to 0-100
      
      // Audio metrics (normalize)
      pitchVariation: this.clamp(raw.pitchVariation, 0, 100),
      volumeNormalized: this.normalizeVolume(raw.volume), // dB to 0-100
      volumeVariation: this.clamp(raw.volumeVariation, 0, 100),
      audioClarity: this.clamp(raw.clarity, 0, 100),
      energy: this.normalizeEnergy(raw.energy),
      
      // Speech metrics (already 0-100)
      wpmScore: this.normalizeWPM(raw.wordsPerMinute),
      fillerScore: 100 - Math.min(100, raw.fillerPercentage * 2), // Less fillers = higher score
      speechClarity: this.clamp(raw.clarityScore, 0, 100),
      fluency: this.clamp(raw.fluencyScore, 0, 100),
      articulation: this.clamp(raw.articulationScore, 0, 100),
    };
  }

  // Step 2: Aggregate into high-level features
  private aggregateFeatures(normalized: Record<string, number>): Record<string, number> {
    return {
      eyeContact: normalized.eyeContact,
      
      posture: this.weightedAverage([
        { value: normalized.postureScore, weight: 0.5 },
        { value: normalized.shoulderAlignment, weight: 0.3 },
        { value: normalized.headPosition, weight: 0.2 },
      ]),
      
      bodyLanguage: this.weightedAverage([
        { value: normalized.gestureVariety, weight: 0.6 },
        { value: normalized.handVisibility, weight: 0.4 },
      ]),
      
      facialExpression: normalized.emotionConfidence,
      
      voiceQuality: this.weightedAverage([
        { value: normalized.volumeNormalized, weight: 0.3 },
        { value: normalized.audioClarity, weight: 0.4 },
        { value: normalized.energy, weight: 0.3 },
      ]),
      
      speechClarity: this.weightedAverage([
        { value: normalized.speechClarity, weight: 0.4 },
        { value: normalized.articulation, weight: 0.3 },
        { value: normalized.fluency, weight: 0.3 },
      ]),
      
      contentEngagement: this.weightedAverage([
        { value: normalized.wpmScore, weight: 0.5 },
        { value: normalized.fillerScore, weight: 0.5 },
      ]),
    };
  }

  // Step 3: Apply context-based weighted fusion
  private applyContextWeights(features: Record<string, number>): number {
    const weights = CONTEXT_WEIGHTS[this.context];
    
    return (
      features.eyeContact * weights.eyeContact +
      features.posture * weights.posture +
      features.bodyLanguage * weights.bodyLanguage +
      features.facialExpression * weights.facialExpression +
      features.voiceQuality * weights.voiceQuality +
      features.speechClarity * weights.speechClarity +
      features.contentEngagement * weights.contentEngagement
    );
  }

  // Step 4: Calculate confidence score
  private calculateConfidence(raw: RawMetrics): number {
    let confidence = 100;
    
    // Reduce confidence if critical metrics are missing or low quality
    if (raw.eyeContact < 10) confidence -= 20; // No face detected
    if (raw.volume < -50) confidence -= 15; // Too quiet
    if (raw.wordsPerMinute === 0) confidence -= 10; // No speech
    if (raw.emotionConfidence < 0.3) confidence -= 10; // Uncertain emotion
    
    return Math.max(0, Math.min(100, confidence));
  }

  // Step 5: Temporal smoothing using moving average
  private applyTemporalSmoothing(current: FusedMetrics): FusedMetrics {
    if (this.history.length === 0) {
      return current;
    }

    const smoothed = { ...current };
    const keys: (keyof FusedMetrics)[] = [
      'eyeContact', 'posture', 'bodyLanguage', 'facialExpression',
      'voiceQuality', 'speechClarity', 'contentEngagement', 'overallScore'
    ];

    for (const key of keys) {
      const values = this.history.map(h => h[key] as number).concat(current[key] as number);
      smoothed[key] = this.movingAverage(values) as any;
    }

    return smoothed;
  }

  // Main fusion method
  fuse(raw: RawMetrics): FusedMetrics {
    // Step 1: Normalize
    const normalized = this.normalizeMetrics(raw);
    
    // Step 2: Aggregate
    const features = this.aggregateFeatures(normalized);
    
    // Step 3: Apply context weights
    const overallScore = this.applyContextWeights(features);
    
    // Step 4: Calculate confidence
    const confidence = this.calculateConfidence(raw);
    
    // Create fused metrics
    const fused: FusedMetrics = {
      eyeContact: this.enforceMinimum(features.eyeContact),
      posture: this.enforceMinimum(features.posture),
      bodyLanguage: this.enforceMinimum(features.bodyLanguage),
      facialExpression: this.enforceMinimum(features.facialExpression),
      voiceQuality: this.enforceMinimum(features.voiceQuality),
      speechClarity: this.enforceMinimum(features.speechClarity),
      contentEngagement: this.enforceMinimum(features.contentEngagement),
      overallScore: this.enforceMinimum(overallScore),
      confidence: confidence,
    };
    
    // Step 5: Apply temporal smoothing
    const smoothed = this.applyTemporalSmoothing(fused);
    
    // Add to history
    this.history.push(smoothed);
    if (this.history.length > this.HISTORY_SIZE) {
      this.history.shift();
    }
    
    return smoothed;
  }

  // Utility methods
  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  private enforceMinimum(value: number): number {
    return Math.max(this.MIN_SCORE, Math.round(value));
  }

  private normalizeVolume(volumeDB: number): number {
    // Convert dB (-60 to 0) to 0-100 scale
    // Optimal range: -40 to -10 dB
    if (volumeDB < -60) return 0;
    if (volumeDB > 0) return 100;
    
    const normalized = ((volumeDB + 60) / 60) * 100;
    return this.clamp(normalized, 0, 100);
  }

  private normalizeEnergy(energy: number): number {
    // Normalize energy (0-200 typical range) to 0-100
    return this.clamp((energy / 200) * 100, 0, 100);
  }

  private normalizeWPM(wpm: number): number {
    // Optimal speaking pace: 120-150 WPM
    // Score decreases if too fast or too slow
    if (wpm === 0) return 0;
    
    if (wpm >= 120 && wpm <= 150) {
      return 100; // Optimal range
    } else if (wpm < 120) {
      // Too slow: scale from 0 to 100 as WPM goes from 0 to 120
      return (wpm / 120) * 100;
    } else {
      // Too fast: decrease score as WPM increases beyond 150
      const penalty = Math.min(50, (wpm - 150) * 0.5);
      return Math.max(50, 100 - penalty);
    }
  }

  private weightedAverage(items: Array<{ value: number; weight: number }>): number {
    const totalWeight = items.reduce((sum, item) => sum + item.weight, 0);
    const weightedSum = items.reduce((sum, item) => sum + item.value * item.weight, 0);
    return weightedSum / totalWeight;
  }

  private movingAverage(values: number[]): number {
    if (values.length === 0) return 0;
    const sum = values.reduce((a, b) => a + b, 0);
    return sum / values.length;
  }

  reset() {
    this.history = [];
  }

  getHistory(): FusedMetrics[] {
    return [...this.history];
  }
}
