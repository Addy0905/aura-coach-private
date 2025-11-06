// Multi-Modal Fusion Algorithm for Persona AI
// Implements weighted fusion, temporal smoothing, and confidence scoring
// Optimized version with real-time score calculation and zero-state handling

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
  eyeContact: number;
  posture: number;
  bodyLanguage: number;
  facialExpression: number;
  voiceQuality: number;
  speechClarity: number;
  contentEngagement: number;
}

// Weight configurations for different contexts
const CONTEXT_WEIGHTS: Record<string, ContextWeights> = {
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
  private context: string = 'presentation';
  private history: FusedMetrics[] = [];
  private readonly HISTORY_SIZE = 10; // For temporal smoothing (last 10 frames)
  private readonly SMOOTHING_FACTOR = 0.7; // Exponential smoothing weight

  setContext(context: string): void {
    if (CONTEXT_WEIGHTS[context]) {
      this.context = context;
    }
  }

  /**
   * Main fusion method - combines multi-modal metrics into unified scores
   * Algorithm steps:
   * 1. Detect zero-state (no voice/face) and return zeros if detected
   * 2. Normalize all metrics to 0-100 scale
   * 3. Aggregate related metrics into high-level features
   * 4. Apply context-based weighted fusion
   * 5. Calculate confidence score based on data quality
   * 6. Apply temporal smoothing for stability
   */
  fuse(raw: RawMetrics): FusedMetrics {
    // Step 1: Check for zero-state (no detection)
    if (this.isZeroState(raw)) {
      return this.createZeroMetrics();
    }

    // Step 2: Normalize all raw metrics to 0-100 scale
    const norm = this.normalizeMetrics(raw);
    
    // Step 3: Aggregate into high-level features
    const features = this.aggregateFeatures(norm);
    
    // Step 4: Apply context-based weighted fusion
    const overallScore = this.applyContextWeights(features);
    
    // Step 5: Calculate confidence score
    const confidence = this.calculateConfidence(raw);
    
    // Create fused metrics object
    const fused: FusedMetrics = {
      eyeContact: Math.round(features.eyeContact),
      posture: Math.round(features.posture),
      bodyLanguage: Math.round(features.bodyLanguage),
      facialExpression: Math.round(features.facialExpression),
      voiceQuality: Math.round(features.voiceQuality),
      speechClarity: Math.round(features.speechClarity),
      contentEngagement: Math.round(features.contentEngagement),
      overallScore: Math.round(overallScore),
      confidence: Math.round(confidence),
    };
    
    // Step 6: Apply temporal smoothing for stability
    const smoothed = this.applyTemporalSmoothing(fused);
    
    // Add to history buffer
    this.history.push(smoothed);
    if (this.history.length > this.HISTORY_SIZE) {
      this.history.shift();
    }
    
    return smoothed;
  }

  /**
   * Detects if no meaningful input is present (no face or voice)
   */
  private isZeroState(raw: RawMetrics): boolean {
    const noFace = raw.eyeContact < 5 && raw.postureScore < 5;
    const noVoice = raw.volume < -55 && raw.wordsPerMinute === 0;
    const noSpeech = raw.wordsPerMinute === 0 && raw.clarityScore < 5;
    
    // Return zero state if both visual and audio are absent
    return (noFace && noVoice) || (noFace && noSpeech);
  }

  /**
   * Returns zero metrics when no input is detected
   */
  private createZeroMetrics(): FusedMetrics {
    return {
      eyeContact: 0,
      posture: 0,
      bodyLanguage: 0,
      facialExpression: 0,
      voiceQuality: 0,
      speechClarity: 0,
      contentEngagement: 0,
      overallScore: 0,
      confidence: 0,
    };
  }

  /**
   * Step 2: Normalize all metrics to 0-100 scale
   * Converts various input ranges to unified scale
   */
  private normalizeMetrics(raw: RawMetrics): Record<string, number> {
    return {
      // Vision metrics (already 0-100, just clamp)
      eyeContact: this.clamp(raw.eyeContact, 0, 100),
      postureScore: this.clamp(raw.postureScore, 0, 100),
      shoulderAlignment: this.clamp(raw.shoulderAlignment, 0, 100),
      headPosition: this.clamp(raw.headPosition, 0, 100),
      gestureVariety: this.clamp(raw.gestureVariety, 0, 100),
      handVisibility: this.clamp(raw.handVisibility, 0, 100),
      emotionConfidence: this.clamp(raw.emotionConfidence * 100, 0, 100),
      
      // Audio metrics (convert to 0-100 scale)
      pitchVariation: this.clamp(raw.pitchVariation, 0, 100),
      volumeNormalized: this.normalizeVolume(raw.volume),
      volumeVariation: this.clamp(raw.volumeVariation, 0, 100),
      audioClarity: this.clamp(raw.clarity, 0, 100),
      energy: this.normalizeEnergy(raw.energy),
      
      // Speech metrics (already 0-100, just clamp)
      wpmScore: this.normalizeWPM(raw.wordsPerMinute),
      fillerScore: this.clamp(100 - (raw.fillerPercentage * 2), 0, 100),
      speechClarity: this.clamp(raw.clarityScore, 0, 100),
      fluency: this.clamp(raw.fluencyScore, 0, 100),
      articulation: this.clamp(raw.articulationScore, 0, 100),
    };
  }

  /**
   * Step 3: Aggregate related metrics into high-level features
   * Uses weighted averaging to combine correlated metrics
   */
  private aggregateFeatures(norm: Record<string, number>): Record<string, number> {
    return {
      eyeContact: norm.eyeContact,
      
      // Posture: combines body position metrics
      posture: (
        norm.postureScore * 0.5 +
        norm.shoulderAlignment * 0.3 +
        norm.headPosition * 0.2
      ),
      
      // Body Language: combines gesture and hand movement
      bodyLanguage: (
        norm.gestureVariety * 0.6 +
        norm.handVisibility * 0.4
      ),
      
      facialExpression: norm.emotionConfidence,
      
      // Voice Quality: combines audio characteristics
      voiceQuality: (
        norm.volumeNormalized * 0.3 +
        norm.audioClarity * 0.4 +
        norm.energy * 0.3
      ),
      
      // Speech Clarity: combines articulation metrics
      speechClarity: (
        norm.speechClarity * 0.4 +
        norm.articulation * 0.3 +
        norm.fluency * 0.3
      ),
      
      // Content Engagement: combines pacing and filler metrics
      contentEngagement: (
        norm.wpmScore * 0.5 +
        norm.fillerScore * 0.5
      ),
    };
  }

  /**
   * Step 4: Apply context-based weighted fusion
   * Different contexts prioritize different aspects of communication
   */
  private applyContextWeights(features: Record<string, number>): number {
    const w = CONTEXT_WEIGHTS[this.context];
    
    return (
      features.eyeContact * w.eyeContact +
      features.posture * w.posture +
      features.bodyLanguage * w.bodyLanguage +
      features.facialExpression * w.facialExpression +
      features.voiceQuality * w.voiceQuality +
      features.speechClarity * w.speechClarity +
      features.contentEngagement * w.contentEngagement
    );
  }

  /**
   * Step 5: Calculate confidence score based on data quality
   * Lower confidence when key inputs are missing or poor quality
   */
  private calculateConfidence(raw: RawMetrics): number {
    let confidence = 100;
    
    // Penalize missing or low-quality inputs
    if (raw.eyeContact < 10) confidence -= 20; // No face detected
    if (raw.volume < -50) confidence -= 15; // Too quiet
    if (raw.wordsPerMinute === 0) confidence -= 10; // No speech
    if (raw.emotionConfidence < 0.3) confidence -= 10; // Uncertain emotion
    if (raw.clarity < 30) confidence -= 10; // Poor audio clarity
    
    return this.clamp(confidence, 0, 100);
  }

  /**
   * Step 6: Temporal smoothing using exponential moving average
   * Reduces jitter while staying responsive to changes
   */
  private applyTemporalSmoothing(current: FusedMetrics): FusedMetrics {
    if (this.history.length === 0) {
      return current;
    }

    const prev = this.history[this.history.length - 1];
    const alpha = this.SMOOTHING_FACTOR;
    const smoothed: FusedMetrics = {
      eyeContact: Math.round(alpha * current.eyeContact + (1 - alpha) * prev.eyeContact),
      posture: Math.round(alpha * current.posture + (1 - alpha) * prev.posture),
      bodyLanguage: Math.round(alpha * current.bodyLanguage + (1 - alpha) * prev.bodyLanguage),
      facialExpression: Math.round(alpha * current.facialExpression + (1 - alpha) * prev.facialExpression),
      voiceQuality: Math.round(alpha * current.voiceQuality + (1 - alpha) * prev.voiceQuality),
      speechClarity: Math.round(alpha * current.speechClarity + (1 - alpha) * prev.speechClarity),
      contentEngagement: Math.round(alpha * current.contentEngagement + (1 - alpha) * prev.contentEngagement),
      overallScore: Math.round(alpha * current.overallScore + (1 - alpha) * prev.overallScore),
      confidence: Math.round(alpha * current.confidence + (1 - alpha) * prev.confidence),
    };

    return smoothed;
  }

  // ========== UTILITY METHODS ==========

  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  /**
   * Normalize volume from dB (-60 to 0) to 0-100 scale
   * Optimal speaking range: -40 to -10 dB
   */
  private normalizeVolume(volumeDB: number): number {
    if (volumeDB < -60) return 0;
    if (volumeDB > 0) return 100;
    
    // Linear mapping: -60 dB = 0, 0 dB = 100
    return ((volumeDB + 60) / 60) * 100;
  }

  /**
   * Normalize energy (typical range 0-200) to 0-100
   */
  private normalizeEnergy(energy: number): number {
    return this.clamp((energy / 200) * 100, 0, 100);
  }

  /**
   * Normalize words per minute with optimal range scoring
   * Optimal: 120-150 WPM (100 score)
   * Too slow: <120 WPM (scaled score)
   * Too fast: >150 WPM (penalized score)
   */
  private normalizeWPM(wpm: number): number {
    if (wpm === 0) return 0;
    
    if (wpm >= 120 && wpm <= 150) {
      return 100; // Optimal range
    } else if (wpm < 120) {
      // Too slow: linear scale from 0 to 100
      return (wpm / 120) * 100;
    } else {
      // Too fast: apply penalty (0.5 point per WPM over 150)
      const penalty = Math.min(50, (wpm - 150) * 0.5);
      return Math.max(50, 100 - penalty);
    }
  }

  /**
   * Reset algorithm state (clears history)
   */
  reset(): void {
    this.history = [];
  }

  /**
   * Get smoothing history (for debugging/analysis)
   */
  getHistory(): FusedMetrics[] {
    return [...this.history];
  }
}

// Export singleton instance for easy use
export const fusionAlgorithm = new FusionAlgorithm();
