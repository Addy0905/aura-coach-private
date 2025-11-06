// Multi-Modal Fusion Algorithm for Persona AI
// Real-time sensor fusion with adaptive weighting and Kalman-inspired smoothing
// Zero-latency processing with intelligent zero-state detection

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

/**
 * ALGORITHM EXPLANATION:
 * 
 * This implements a MULTI-MODAL SENSOR FUSION ALGORITHM with:
 * 
 * 1. WEIGHTED FUSION - Combines vision, audio, and speech data
 *    - Each modality contributes based on context (professional/presentation/casual)
 *    - Uses weighted averaging to merge correlated metrics
 * 
 * 2. KALMAN-INSPIRED SMOOTHING - Reduces noise while staying responsive
 *    - Exponential moving average (EMA) with adaptive alpha
 *    - Balances current measurement vs historical trend
 *    - Formula: smoothed = α * current + (1-α) * previous
 * 
 * 3. CONFIDENCE SCORING - Quality assessment of input data
 *    - Bayesian-style confidence reduction for missing/poor inputs
 *    - Helps UI know when scores are unreliable
 * 
 * 4. ZERO-STATE DETECTION - Intelligent null detection
 *    - Returns 0 when no face/voice detected (not artificial minimums)
 *    - Prevents false scores from noise
 * 
 * 5. REAL-TIME NORMALIZATION - Converts all metrics to 0-100 scale
 *    - Handles different sensor ranges (dB, Hz, percentages)
 *    - Non-linear curves for WPM (optimal at 120-150)
 */
export class FusionAlgorithm {
  private context: string = 'presentation';
  private previousMetrics: FusedMetrics | null = null;
  private readonly SMOOTHING_ALPHA = 0.7; // Higher = more responsive (0.7 = 70% current, 30% history)
  private readonly CONFIDENCE_THRESHOLD = 40; // Below this, scores are unreliable

  setContext(context: string): void {
    if (CONTEXT_WEIGHTS[context]) {
      this.context = context;
    }
  }

  /**
   * MAIN FUSION PIPELINE
   * Processes raw sensor data into unified performance scores
   * 
   * Pipeline: Raw Input → Zero Check → Normalize → Aggregate → Weight → Smooth → Output
   */
  fuse(raw: RawMetrics): FusedMetrics {
    // STEP 1: ZERO-STATE DETECTION
    // Check if no meaningful data is present (no face AND no voice)
    if (this.isZeroState(raw)) {
      const zeros = this.createZeroMetrics();
      this.previousMetrics = zeros;
      return zeros;
    }

    // STEP 2: NORMALIZATION (all metrics → 0-100 scale)
    const norm = this.normalizeMetrics(raw);
    
    // STEP 3: FEATURE AGGREGATION (combine related metrics)
    const features = this.aggregateFeatures(norm);
    
    // STEP 4: CONTEXT-BASED WEIGHTING (apply importance weights)
    const overallScore = this.applyContextWeights(features);
    
    // STEP 5: CONFIDENCE CALCULATION (data quality assessment)
    const confidence = this.calculateConfidence(raw);
    
    // Create current frame metrics
    const current: FusedMetrics = {
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
    
    // STEP 6: TEMPORAL SMOOTHING (Kalman-inspired EMA)
    const smoothed = this.applyAdaptiveSmoothing(current, confidence);
    
    // Store for next iteration
    this.previousMetrics = smoothed;
    
    return smoothed;
  }

  /**
   * ZERO-STATE DETECTION
   * Returns true if no meaningful input is detected
   * Prevents false scores from sensor noise
   */
  private isZeroState(raw: RawMetrics): boolean {
    const noFace = raw.eyeContact < 5 && raw.postureScore < 5;
    const noAudio = raw.volume < -55 && raw.energy < 5;
    const noSpeech = raw.wordsPerMinute === 0 && raw.clarityScore < 5;
    
    // Zero state = (no face AND no audio) OR (no face AND no speech)
    return (noFace && noAudio) || (noFace && noSpeech);
  }

  /**
   * Creates zero metrics when no input detected
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
   * STEP 2: NORMALIZATION
   * Converts all sensor readings to unified 0-100 scale
   * Handles: dB ranges, Hz frequencies, percentages, confidences
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
      
      // Audio metrics (convert to 0-100)
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
   * STEP 3: FEATURE AGGREGATION
   * Combines correlated metrics using weighted averaging
   * Reduces dimensionality: 14 metrics → 7 features
   */
  private aggregateFeatures(norm: Record<string, number>): Record<string, number> {
    return {
      eyeContact: norm.eyeContact,
      
      // Posture = body position + alignment
      posture: (
        norm.postureScore * 0.5 +
        norm.shoulderAlignment * 0.3 +
        norm.headPosition * 0.2
      ),
      
      // Body Language = gestures + hand movement
      bodyLanguage: (
        norm.gestureVariety * 0.6 +
        norm.handVisibility * 0.4
      ),
      
      facialExpression: norm.emotionConfidence,
      
      // Voice Quality = volume + clarity + energy
      voiceQuality: (
        norm.volumeNormalized * 0.3 +
        norm.audioClarity * 0.4 +
        norm.energy * 0.3
      ),
      
      // Speech Clarity = articulation + fluency
      speechClarity: (
        norm.speechClarity * 0.4 +
        norm.articulation * 0.3 +
        norm.fluency * 0.3
      ),
      
      // Content Engagement = pacing + filler reduction
      contentEngagement: (
        norm.wpmScore * 0.5 +
        norm.fillerScore * 0.5
      ),
    };
  }

  /**
   * STEP 4: CONTEXT-BASED WEIGHTED FUSION
   * Applies different importance to features based on scenario
   * 
   * Example: Presentation = 25% eye contact, 15% posture
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
   * STEP 5: CONFIDENCE CALCULATION
   * Bayesian-inspired quality assessment
   * Starts at 100%, applies penalties for poor/missing data
   */
  private calculateConfidence(raw: RawMetrics): number {
    let confidence = 100;
    
    // Apply penalties for missing or low-quality inputs
    if (raw.eyeContact < 10) confidence -= 20; // No face detected
    if (raw.volume < -50) confidence -= 15; // Too quiet
    if (raw.wordsPerMinute === 0) confidence -= 10; // No speech
    if (raw.emotionConfidence < 0.3) confidence -= 10; // Uncertain emotion
    if (raw.clarity < 30) confidence -= 10; // Poor audio clarity
    if (raw.postureScore < 20) confidence -= 10; // Poor posture detection
    
    return this.clamp(confidence, 0, 100);
  }

  /**
   * STEP 6: ADAPTIVE TEMPORAL SMOOTHING
   * Kalman-inspired exponential moving average (EMA)
   * 
   * Formula: smoothed = α * current + (1-α) * previous
   * - α = SMOOTHING_ALPHA (0.7 = 70% current, 30% history)
   * - Adapts based on confidence: low confidence = more smoothing
   * 
   * Benefits:
   * - Reduces jitter/flickering
   * - Stays responsive to real changes
   * - Lower latency than moving average
   */
  private applyAdaptiveSmoothing(current: FusedMetrics, confidence: number): FusedMetrics {
    if (!this.previousMetrics) {
      return current; // First frame, no history
    }

    // Adapt smoothing based on confidence
    // Low confidence = smooth more (reduce noise)
    // High confidence = smooth less (stay responsive)
    const adaptiveAlpha = confidence > this.CONFIDENCE_THRESHOLD 
      ? this.SMOOTHING_ALPHA 
      : this.SMOOTHING_ALPHA * 0.6; // Smooth more when uncertain

    const prev = this.previousMetrics;
    
    return {
      eyeContact: Math.round(adaptiveAlpha * current.eyeContact + (1 - adaptiveAlpha) * prev.eyeContact),
      posture: Math.round(adaptiveAlpha * current.posture + (1 - adaptiveAlpha) * prev.posture),
      bodyLanguage: Math.round(adaptiveAlpha * current.bodyLanguage + (1 - adaptiveAlpha) * prev.bodyLanguage),
      facialExpression: Math.round(adaptiveAlpha * current.facialExpression + (1 - adaptiveAlpha) * prev.facialExpression),
      voiceQuality: Math.round(adaptiveAlpha * current.voiceQuality + (1 - adaptiveAlpha) * prev.voiceQuality),
      speechClarity: Math.round(adaptiveAlpha * current.speechClarity + (1 - adaptiveAlpha) * prev.speechClarity),
      contentEngagement: Math.round(adaptiveAlpha * current.contentEngagement + (1 - adaptiveAlpha) * prev.contentEngagement),
      overallScore: Math.round(adaptiveAlpha * current.overallScore + (1 - adaptiveAlpha) * prev.overallScore),
      confidence: Math.round(adaptiveAlpha * current.confidence + (1 - adaptiveAlpha) * prev.confidence),
    };
  }

  // ========== UTILITY METHODS ==========

  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  /**
   * VOLUME NORMALIZATION
   * Converts dB (-60 to 0) → 0-100 scale
   * Linear mapping: -60 dB = silence (0), 0 dB = max (100)
   * Optimal speaking: -40 to -10 dB
   */
  private normalizeVolume(volumeDB: number): number {
    if (volumeDB < -60) return 0;
    if (volumeDB > 0) return 100;
    return ((volumeDB + 60) / 60) * 100;
  }

  /**
   * ENERGY NORMALIZATION
   * Typical audio energy range: 0-200 → 0-100 scale
   */
  private normalizeEnergy(energy: number): number {
    return this.clamp((energy / 200) * 100, 0, 100);
  }

  /**
   * WPM NORMALIZATION WITH OPTIMAL CURVE
   * Non-linear scoring:
   * - 120-150 WPM = 100 (optimal speaking pace)
   * - <120 WPM = scaled down (too slow)
   * - >150 WPM = penalized (too fast)
   */
  private normalizeWPM(wpm: number): number {
    if (wpm === 0) return 0;
    
    if (wpm >= 120 && wpm <= 150) {
      return 100; // Optimal range
    } else if (wpm < 120) {
      // Too slow: linear scale 0-120 → 0-100
      return (wpm / 120) * 100;
    } else {
      // Too fast: penalty of 0.5 per WPM over 150
      const penalty = Math.min(50, (wpm - 150) * 0.5);
      return Math.max(50, 100 - penalty);
    }
  }

  /**
   * Reset algorithm state (clears history)
   */
  reset(): void {
    this.previousMetrics = null;
  }

  /**
   * Get current smoothing state (for debugging)
   */
  getPreviousMetrics(): FusedMetrics | null {
    return this.previousMetrics ? { ...this.previousMetrics } : null;
  }
}

// Export singleton instance for easy usage
export const fusionAlgorithm = new FusionAlgorithm();
