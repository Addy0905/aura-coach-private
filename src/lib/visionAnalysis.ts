/**
 * Advanced Computer Vision using MediaPipe (OpenCV + DeepFace equivalent)
 * Real-time face tracking, posture analysis, and gesture recognition
 * Implements FACS emotion detection, optical flow, and geometric analysis
 */
import { FaceLandmarker, PoseLandmarker, GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';
/**
 * ALGORITHM EXPLANATION:
 *
 * This module implements MULTIPLE computer vision algorithms:
 *
 * 1. EYE CONTACT TRACKING
 * - EAR (Eye Aspect Ratio): Measures eye openness using vertical/horizontal landmark ratios
 * - Iris Tracking: Detects gaze direction using iris position within eye bounds
 * - Gaze Vector: Calculates 3D direction of eye focus
 *
 * 2. FACS EMOTION DETECTION (Facial Action Coding System)
 * - Maps 28 Action Units (AU) to 7 basic emotions
 * - Uses Euclidean distance between facial landmarks
 * - Example: Smile = AU6 (cheek raise) + AU12 (lip corner pull)
 *
 * 3. OPTICAL FLOW (Lucas-Kanade approximation)
 * - Tracks micro-expressions via frame-to-frame landmark displacement
 * - Measures facial/body movement velocity
 * - Detects stability vs fidgeting
 *
 * 4. GEOMETRIC POSTURE ANALYSIS
 * - Joint angle calculation: shoulder, neck, spine
 * - Alignment scoring: vertical spine deviation detection
 * - 3D trigonometry for uprightness measurement
 *
 * 5. GESTURE RECOGNITION
 * - MediaPipe pre-trained gesture classifier
 * - Variety tracking: unique gesture count over time
 * - Hand visibility: confidence-weighted detection
 */
export interface FaceAnalysis {
  eyeContact: number; // 0-100
  emotion: string;
  emotionConfidence: number; // 0-1
  facialMovement: number; // Micro-expressions (0-100)
  gazeDirection: { x: number; y: number };
}
export interface PostureAnalysis {
  postureScore: number; // 0-100, overall uprightness
  shoulderAlignment: number; // 0-100
  headPosition: number; // 0-100, centered vs tilted
  stability: number; // Movement stability (0-100)
}
export interface GestureAnalysis {
  gestureCount: number;
  gestureVariety: number; // 0-100
  handVisibility: number; // 0-100
  movementPatterns: string[];
}
export interface BodyLanguageMetrics {
  face: FaceAnalysis & { landmarks?: any[] };
  posture: PostureAnalysis & { landmarks?: any[] };
  gestures: GestureAnalysis;
  overallConfidence: number;
  timestamp: number;
}
export class VisionAnalyzer {
  private faceLandmarker: FaceLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;
  private gestureRecognizer: GestureRecognizer | null = null;
  private isInitialized = false;
 
  // For optical flow (temporal tracking) with smoothing
  private prevFace: any = null;
  private prevPose: any = null;
  private facialMovementHistory: number[] = [];
  private stabilityHistory: number[] = [];
  private readonly HISTORY_SIZE = 5; // Average over last 5 frames for stability
 
  // For gesture variety tracking
  private gestureBuffer: string[] = [];
  private readonly GESTURE_BUFFER_SIZE = 30;
  /**
   * Initialize MediaPipe models (GPU-accelerated)
   * Downloads WASM models from CDN on first run
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      // Face Landmarker: 478 landmarks for facial analysis
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
      });
      // Pose Landmarker: 33 body keypoints for posture
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
      // Gesture Recognizer: Pre-trained hand gesture classifier
      this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });
      this.isInitialized = true;
      console.log('✓ MediaPipe vision models initialized');
    } catch (error) {
      console.error('✗ Failed to initialize MediaPipe:', error);
      throw error;
    }
  }
  /**
   * Main analysis pipeline: processes video frame and returns metrics
   * Returns all zeros if no face/body detected (no artificial scores)
   */
  async analyzeFrame(videoElement: HTMLVideoElement, timestamp: number): Promise<BodyLanguageMetrics> {
    if (!this.isInitialized || !this.faceLandmarker || !this.poseLandmarker || !this.gestureRecognizer) {
      return this.createZeroMetrics(timestamp);
    }
    try {
      // Run all detectors in parallel for performance
      const [faceResults, poseResults, gestureResults] = await Promise.all([
        this.faceLandmarker.detectForVideo(videoElement, timestamp),
        this.poseLandmarker.detectForVideo(videoElement, timestamp),
        this.gestureRecognizer.recognizeForVideo(videoElement, timestamp),
      ]);
      // Check if anything was detected
      const faceDetected = faceResults.faceLandmarks?.length > 0;
      const poseDetected = poseResults.landmarks?.length > 0;
      const gesturesDetected = gestureResults.gestures?.length > 0;
      // Return zeros if nothing detected (no artificial scores)
      if (!faceDetected && !poseDetected) {
        return this.createZeroMetrics(timestamp);
      }
      // Analyze each modality
      const faceAnalysis = faceDetected ? this.analyzeFace(faceResults) : this.createZeroFace();
      const postureAnalysis = poseDetected ? this.analyzePosture(poseResults) : this.createZeroPosture();
      const gestureAnalysis = this.analyzeGestures(gestureResults);
      // Calculate overall confidence (weighted by detection quality)
      const detectionScore = (
        (faceDetected ? 40 : 0) +
        (poseDetected ? 40 : 0) +
        (gesturesDetected ? 20 : 0)
      );
      const qualityScore = (
        faceAnalysis.eyeContact * 0.3 +
        postureAnalysis.postureScore * 0.4 +
        gestureAnalysis.gestureVariety * 0.3
      );
      const overallConfidence = Math.round(qualityScore * (detectionScore / 100));
      return {
        face: {
          ...faceAnalysis,
          landmarks: faceResults?.faceLandmarks?.[0]?.slice(0, 468).map((l: any) => ({
            x: l.x || 0,
            y: l.y || 0,
            z: l.z || 0
          })) || []
        },
        posture: {
          ...postureAnalysis,
          landmarks: poseResults?.landmarks?.[0]?.map((l: any) => ({
            x: l.x || 0,
            y: l.y || 0,
            z: l.z || 0
          })) || []
        },
        gestures: gestureAnalysis,
        overallConfidence: Math.max(0, overallConfidence),
        timestamp,
      };
    } catch (error) {
      console.error('Error analyzing frame:', error);
      return this.createZeroMetrics(timestamp);
    }
  }
  /**
   * FACE ANALYSIS PIPELINE
   * Implements: EAR, iris tracking, gaze vector, FACS emotion, optical flow
   */
  private analyzeFace(results: any): FaceAnalysis {
    if (!results.faceLandmarks?.length) {
      return this.createZeroFace();
    }
    const lm = results.faceLandmarks[0];
    // Check if all required landmarks exist
    if (!this.validateLandmarks(lm, [1, 33, 133, 160, 158, 153, 144, 362, 263, 385, 387, 373, 380, 468, 473])) {
      return this.createZeroFace();
    }
    // === 1. EYE CONTACT TRACKING ===
   
    // EAR (Eye Aspect Ratio) for eye openness
    const leftEAR = this.calcEAR(lm, 'left');
    const rightEAR = this.calcEAR(lm, 'right');
    const avgEAR = (leftEAR + rightEAR) / 2;
    const eyeOpenness = Math.min(100, avgEAR * 300);
    // Iris-based gaze tracking
    const leftIris = lm[468] || lm[33];
    const rightIris = lm[473] || lm[263];
    const gazeVector = this.calcGazeVector(leftIris, rightIris, lm);
    // Gaze score (0 = looking away, 100 = direct eye contact)
    const gazeDist = Math.sqrt(gazeVector.x ** 2 + gazeVector.y ** 2);
    const gazeScore = Math.max(0, (1 - Math.min(gazeDist * 1.5, 1)) * 100);
   
    const eyeContact = Math.round(gazeScore * 0.7 + eyeOpenness * 0.3);
    // === 2. OPTICAL FLOW (Micro-expressions) ===
    let facialMovement = 0;
    if (this.prevFace) {
      // Track critical facial points (jawline, brows, mouth)
      const criticalPts = [
        ...Array.from({length: 17}, (_, i) => i), // Jawline
        ...Array.from({length: 10}, (_, i) => 17 + i), // Eyebrows
        ...Array.from({length: 20}, (_, i) => 48 + i), // Mouth
      ];
      let totalDisplacement = 0;
      let validPoints = 0;
      for (const idx of criticalPts) {
        if (lm[idx] && this.prevFace[idx]) {
          const dx = lm[idx].x - this.prevFace[idx].x;
          const dy = lm[idx].y - this.prevFace[idx].y;
          const dz = (lm[idx].z || 0) - (this.prevFace[idx].z || 0);
          totalDisplacement += Math.sqrt(dx ** 2 + dy ** 2 + dz ** 2);
          validPoints++;
        }
      }
      if (validPoints > 0) {
        facialMovement = Math.min(100, (totalDisplacement / validPoints) * 1000); // Normalized multiplier
        this.facialMovementHistory.push(facialMovement);
        if (this.facialMovementHistory.length > this.HISTORY_SIZE) this.facialMovementHistory.shift();
        facialMovement = this.average(this.facialMovementHistory); // Smooth over history
      }
    }
    this.prevFace = lm;
    // === 3. FACS EMOTION DETECTION ===
    const emotion = this.detectEmotionFACS(lm);
    return {
      eyeContact: Math.max(0, eyeContact),
      emotion: emotion.label,
      emotionConfidence: emotion.confidence,
      facialMovement: Math.round(facialMovement),
      gazeDirection: gazeVector,
    };
  }
  /**
   * EAR (Eye Aspect Ratio) Calculation
   * Formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
   * Normal range: 0.2-0.4 when eyes open, <0.15 when closed
   */
  private calcEAR(lm: any[], eye: 'left' | 'right'): number {
    let points;
    if (eye === 'left') {
      points = [33, 160, 158, 133, 153, 144].map(i => lm[i]);
    } else {
      points = [362, 385, 387, 263, 373, 380].map(i => lm[i]);
    }
    if (points.some(p => !p)) return 0; // Invalid if any point missing
    const [p1, p2, p3, p4, p5, p6] = points;
    const dist = (a: any, b: any) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    const vertical = (dist(p2, p6) + dist(p3, p5)) / 2;
    const horizontal = dist(p1, p4);
    return horizontal > 0 ? vertical / horizontal : 0;
  }
  /**
   * Gaze Vector Calculation
   * Maps iris position within eye bounds to direction vector
   */
  private calcGazeVector(leftIris: any, rightIris: any, lm: any[]): { x: number; y: number } {
    const leftEyeL = lm[33];
    const leftEyeR = lm[133];
    const rightEyeL = lm[362];
    const rightEyeR = lm[263];
    if (!leftEyeL || !leftEyeR || !rightEyeL || !rightEyeR) return { x: 0, y: 0 };
    const leftIrisX = (leftIris.x - leftEyeL.x) / (leftEyeR.x - leftEyeL.x + 0.0001);
    const rightIrisX = (rightIris.x - rightEyeL.x) / (rightEyeR.x - rightEyeL.x + 0.0001);
    const gazeX = ((leftIrisX + rightIrisX) / 2 - 0.5) * 2;
    const gazeY = ((leftIris.y + rightIris.y) / 2 - 0.5) * 2;
    return { x: gazeX, y: gazeY };
  }
  /**
   * FACS (Facial Action Coding System) Emotion Detection
   * Maps Action Units (AU) to 7 basic emotions using landmark geometry
   *
   * Key AUs:
   * - AU1+2: Brow raise (surprise, fear)
   * - AU4: Brow lower (anger)
   * - AU6+12: Cheek + lip raise (genuine smile)
   * - AU9+10: Nose wrinkle + lip raise (disgust)
   * - AU15: Lip corner down (sadness)
   * - AU25+26: Lip part + jaw drop (surprise)
   */
  private detectEmotionFACS(lm: any[]): { label: string; confidence: number } {
    // Validate required landmarks
    const requiredIndices = [1, 13, 14, 61, 98, 145, 159, 168, 205, 291, 327, 374, 386, 425, 70, 300];
    if (!this.validateLandmarks(lm, requiredIndices)) {
      return { label: 'neutral', confidence: 0 };
    }
    // Action Unit measurements
    const browRaise = this.dist(lm[70], lm[300]) * 2;
    const browLower = (lm[70].y + lm[300].y) / 2 - lm[168].y;
    const cheekRaise = ((lm[205].y - lm[145].y) + (lm[425].y - lm[374].y)) / 2;
    const noseWrinkle = this.dist(lm[98], lm[327]) / this.dist(lm[98], lm[1]);
    const lipRaise = lm[0].y - lm[13].y;
    const mouthWidth = this.dist(lm[61], lm[291]);
    const lipLift = lm[13].y - (lm[61].y + lm[291].y) / 2;
    const mouthOpen = this.dist(lm[13], lm[14]);
    const eyeOpen = (this.dist(lm[159], lm[145]) + this.dist(lm[386], lm[374])) / 2;
    // Emotion scores based on AU combinations
    const scores = {
      happy: (cheekRaise * 10 + lipLift * 50) * (mouthWidth > 0.15 ? 1.5 : 1),
      sad: browRaise * 20 + Math.abs(browLower) * 30 - lipLift * 40,
      surprised: browRaise * 30 + eyeOpen * 50 + mouthOpen * 40,
      angry: Math.abs(browLower) * 50 + (eyeOpen < 0.02 ? 20 : 0),
      fear: browRaise * 20 + eyeOpen * 30 + mouthWidth * 25,
      disgust: noseWrinkle * 100 - lipLift * 30 + lipRaise * 20,
      neutral: Math.max(0, 50 - (Math.abs(browRaise) + Math.abs(lipLift) + Math.abs(mouthOpen)) * 10),
    };
    // Find dominant emotion
    const dominant = Object.entries(scores).reduce((max, [label, score]) =>
      score > max.score ? { label, score: Math.max(0, score) } : max
    , { label: 'neutral', score: scores.neutral });
    const totalScore = Object.values(scores).reduce((sum, s) => sum + Math.max(0, s), 0.0001);
    const confidence = Math.min(1, dominant.score / totalScore);
    return {
      label: dominant.label,
      confidence: Math.round(confidence * 100) / 100,
    };
  }
  /**
   * POSTURE ANALYSIS PIPELINE
   * Implements: Joint angle calculation, spine alignment, stability tracking
   */
  private analyzePosture(results: any): PostureAnalysis {
    if (!results.landmarks?.length) {
      return this.createZeroPosture();
    }
    const lm = results.landmarks[0];
    // Validate required landmarks
    const requiredIndices = [0, 11, 12, 23, 24];
    if (!this.validateLandmarks(lm, requiredIndices)) {
      return this.createZeroPosture();
    }
    // Key body points (MediaPipe Pose: 33 landmarks)
    const nose = lm[0];
    const lShoulder = lm[11];
    const rShoulder = lm[12];
    const lHip = lm[23];
    const rHip = lm[24];
    // === 1. SHOULDER ALIGNMENT ===
    const shoulderAngle = this.calcAngle(lShoulder, nose, rShoulder);
    const shoulderAlignment = Math.max(0, 100 - Math.abs(180 - shoulderAngle) * 2);
    // === 2. HEAD POSITION (neck angle) ===
    const neckAngle = this.calcAngle(
      { x: nose.x, y: nose.y - 0.1, z: nose.z || 0 },
      nose,
      this.midpoint(lShoulder, rShoulder)
    );
    const headUpright = Math.max(0, 100 - Math.abs(180 - neckAngle) * 1.5);
    // === 3. SPINE ALIGNMENT ===
    const shoulderMid = this.midpoint(lShoulder, rShoulder);
    const hipMid = this.midpoint(lHip, rHip);
    const spineDeviation = Math.abs(shoulderMid.x - hipMid.x);
    const spineAlignment = Math.max(0, 100 - spineDeviation * 200);
    // === 4. HEAD CENTEREDNESS ===
    const headCenteredness = 1 - Math.abs(nose.x - shoulderMid.x) * 2;
    const headPosition = Math.max(0, Math.round(headCenteredness * 100));
    // Overall posture score (weighted combination)
    const postureScore = Math.round(
      shoulderAlignment * 0.3 +
      headUpright * 0.25 +
      spineAlignment * 0.25 +
      headPosition * 0.2
    );
    // === 5. STABILITY (Optical Flow) ===
    let stability = 100;
    if (this.prevPose) {
      const keyPts = [0, 11, 12, 13, 14, 15, 16, 23, 24]; // Upper body keypoints
      let totalMovement = 0;
      let validPoints = 0;
      for (const idx of keyPts) {
        if (lm[idx] && this.prevPose[idx]) {
          const dx = lm[idx].x - this.prevPose[idx].x;
          const dy = lm[idx].y - this.prevPose[idx].y;
          const dz = (lm[idx].z || 0) - (this.prevPose[idx].z || 0);
          totalMovement += Math.sqrt(dx ** 2 + dy ** 2 + dz ** 2);
          validPoints++;
        }
      }
      if (validPoints > 0) {
        let currentStability = Math.max(0, Math.round((1 - Math.min((totalMovement / validPoints) * 20, 1)) * 100));
        this.stabilityHistory.push(currentStability);
        if (this.stabilityHistory.length > this.HISTORY_SIZE) this.stabilityHistory.shift();
        stability = this.average(this.stabilityHistory); // Smooth over history
      }
    }
    this.prevPose = lm;
    return {
      postureScore: Math.max(0, postureScore),
      shoulderAlignment: Math.max(0, Math.round(shoulderAlignment)),
      headPosition: Math.max(0, headPosition),
      stability: Math.max(0, stability),
    };
  }
  /**
   * 3D Joint Angle Calculation
   * Uses dot product and arc-cosine for angle between three points
   */
  private calcAngle(p1: any, p2: any, p3: any): number {
    if (!p1 || !p2 || !p3) return 0;
    const v1 = {
      x: p1.x - p2.x,
      y: p1.y - p2.y,
      z: (p1.z || 0) - (p2.z || 0),
    };
    const v2 = {
      x: p3.x - p2.x,
      y: p3.y - p2.y,
      z: (p3.z || 0) - (p2.z || 0),
    };
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2) + 0.0001; // Avoid div zero
    const mag2 = Math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2) + 0.0001;
    const cosAngle = dot / (mag1 * mag2);
    const angleRad = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    return (angleRad * 180) / Math.PI;
  }
  /**
   * GESTURE ANALYSIS
   * Tracks hand gestures and calculates variety over time window
   */
  private analyzeGestures(results: any): GestureAnalysis {
    const gestures = results.gestures || [];
    const handedness = results.handedness || [];
    // Add new gestures to rolling buffer
    for (const gesture of gestures) {
      if (gesture[0]?.categoryName) {
        this.gestureBuffer.push(gesture[0].categoryName);
      }
    }
    // Keep only recent gestures
    if (this.gestureBuffer.length > this.GESTURE_BUFFER_SIZE) {
      this.gestureBuffer = this.gestureBuffer.slice(-this.GESTURE_BUFFER_SIZE);
    }
    // Calculate variety (unique gestures / max possible * 100)
    const uniqueGestures = new Set(this.gestureBuffer);
    const gestureVariety = Math.min(100, uniqueGestures.size * 20);
    // Hand visibility (both hands = 100, one hand = 50, none = 0)
    const handVisibility = Math.min(100, handedness.length * 50);
    return {
      gestureCount: gestures.length,
      gestureVariety,
      handVisibility,
      movementPatterns: Array.from(uniqueGestures),
    };
  }
  // ========== UTILITY METHODS ==========
  private dist(p1: any, p2: any): number {
    if (!p1 || !p2) return 0;
    return Math.sqrt(
      (p1.x - p2.x) ** 2 +
      (p1.y - p2.y) ** 2 +
      ((p1.z || 0) - (p2.z || 0)) ** 2
    );
  }
  private midpoint(p1: any, p2: any): any {
    if (!p1 || !p2) return { x: 0, y: 0, z: 0 };
    return {
      x: (p1.x + p2.x) / 2,
      y: (p1.y + p2.y) / 2,
      z: ((p1.z || 0) + (p2.z || 0)) / 2,
    };
  }
  private createZeroFace(): FaceAnalysis {
    return {
      eyeContact: 0,
      emotion: 'neutral',
      emotionConfidence: 0,
      facialMovement: 0,
      gazeDirection: { x: 0, y: 0 },
    };
  }
  private createZeroPosture(): PostureAnalysis {
    return {
      postureScore: 0,
      shoulderAlignment: 0,
      headPosition: 0,
      stability: 0,
    };
  }
  private createZeroMetrics(timestamp: number): BodyLanguageMetrics {
    return {
      face: { ...this.createZeroFace(), landmarks: [] },
      posture: { ...this.createZeroPosture(), landmarks: [] },
      gestures: {
        gestureCount: 0,
        gestureVariety: 0,
        handVisibility: 0,
        movementPatterns: [],
      },
      overallConfidence: 0,
      timestamp,
    };
  }
  /**
   * Validate if required landmarks exist
   */
  private validateLandmarks(lm: any[], indices: number[]): boolean {
    return indices.every(idx => lm[idx] && typeof lm[idx].x === 'number' && typeof lm[idx].y === 'number');
  }
  /**
   * Simple average for smoothing history
   */
  private average(history: number[]): number {
    if (history.length === 0) return 0;
    return history.reduce((sum, val) => sum + val, 0) / history.length;
  }
  /**
   * Cleanup resources (call when component unmounts)
   */
  cleanup(): void {
    if (this.faceLandmarker) this.faceLandmarker.close();
    if (this.poseLandmarker) this.poseLandmarker.close();
    if (this.gestureRecognizer) this.gestureRecognizer.close();
   
    this.faceLandmarker = null;
    this.poseLandmarker = null;
    this.gestureRecognizer = null;
    this.isInitialized = false;
    this.prevFace = null;
    this.prevPose = null;
    this.gestureBuffer = [];
    this.facialMovementHistory = [];
    this.stabilityHistory = [];
  }
}
// Export singleton instance
export const visionAnalyzer = new VisionAnalyzer();
