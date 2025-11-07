```typescript
// Advanced computer vision and audio analysis using MediaPipe with optimized FACS emotion detection and YIN pitch detection
import { FaceLandmarker, PoseLandmarker, GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';

export interface FaceAnalysis {
  eyeContact: number; // 0-100
  emotion: string;
  emotionConfidence: number;
  facialMovement: number; // Micro-expressions
  gazeDirection: { x: number; y: number };
  emotionScores?: Record<string, number>; // All emotion scores for debugging
}

export interface PostureAnalysis {
  postureScore: number; // 0-100, uprightness
  shoulderAlignment: number; // 0-100
  headPosition: number; // 0-100, centered vs tilted
  stability: number; // Movement stability
}

export interface GestureAnalysis {
  gestureCount: number;
  gestureVariety: number; // 0-100
  handVisibility: number; // 0-100
  movementPatterns: string[];
}

export interface AudioAnalysis {
  pitch: number; // Fundamental frequency in Hz (using YIN algorithm)
  volume: number; // RMS volume (0-100)
  speechRate: number; // Approximate words per minute (simple estimation)
  voiceStability: number; // Pitch variation (0-100, lower is more stable)
  audioEmotion: string; // Basic emotion inference from pitch/volume (e.g., excited, calm)
  audioConfidence: number; // 0-1
}

export interface BodyLanguageMetrics {
  face: FaceAnalysis & { landmarks?: any[] };
  posture: PostureAnalysis & { landmarks?: any[] };
  gestures: GestureAnalysis;
  audio: AudioAnalysis;
  overallConfidence: number;
  timestamp: number;
}

export class VisionAnalyzer {
  private faceLandmarker: FaceLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;
  private gestureRecognizer: GestureRecognizer | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private mediaSource: MediaElementSourceNode | null = null;
  private isInitialized = false;
  private previousFaceLandmarks: any = null;
  private previousPoseLandmarks: any = null;
  private gestureHistory: string[] = [];
  private emotionHistory: string[] = [];
  private pitchHistory: number[] = [];
  private frameCount = 0;

  // Calibrated thresholds for better detection
  private readonly EAR_THRESHOLD = 0.21; // Eye aspect ratio for closed eyes
  private readonly BLINK_FRAMES = 2; // Frames to confirm blink
  private readonly EMOTION_SMOOTHING = 5; // Frames to smooth emotion changes
  private readonly MOVEMENT_THRESHOLD = 0.002; // Threshold for micro-expressions
  private readonly SAMPLE_RATE = 44100; // Standard audio sample rate
  private readonly FFT_SIZE = 2048; // For audio analysis
  private readonly YIN_THRESHOLD = 0.1; // For YIN pitch detection
  private readonly ENERGY_THRESHOLD = 0.02; // RMS threshold for speech
  private readonly ZCR_MIN = 0.1; // Min zero-crossing rate for speech
  private readonly ZCR_MAX = 0.4; // Max zero-crossing rate to filter noise

  async initialize(videoElement?: HTMLVideoElement) {
    if (this.isInitialized) return;
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      // Initialize Face Landmarker with optimized settings
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: true, // Enable blendshapes for better emotion detection
        outputFacialTransformationMatrixes: true,
      });
      // Initialize Pose Landmarker
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      // Initialize Gesture Recognizer
      this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      // Initialize Audio Context if videoElement provided
      if (videoElement) {
        this.audioContext = new AudioContext({ sampleRate: this.SAMPLE_RATE });
        this.mediaSource = this.audioContext.createMediaElementSource(videoElement);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = this.FFT_SIZE;
        this.mediaSource.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);
        await this.audioContext.resume();
      }
      this.isInitialized = true;
      console.log('✓ MediaPipe vision and audio models initialized with optimized settings');
    } catch (error) {
      console.error('Failed to initialize MediaPipe models or audio:', error);
      throw error;
    }
  }

  async analyzeFrame(videoElement: HTMLVideoElement, timestamp: number): Promise<BodyLanguageMetrics> {
    if (!this.isInitialized || !this.faceLandmarker || !this.poseLandmarker || !this.gestureRecognizer) {
      return this.getDefaultMetrics();
    }
    // Ensure audio is set up if not already
    if (!this.audioContext && videoElement) {
      this.audioContext = new AudioContext({ sampleRate: this.SAMPLE_RATE });
      this.mediaSource = this.audioContext.createMediaElementSource(videoElement);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = this.FFT_SIZE;
      this.mediaSource.connect(this.analyser);
      this.analyser.connect(this.audioContext.destination);
      await this.audioContext.resume();
    }
    try {
      this.frameCount++;
      // Analyze face with blendshapes
      const faceResults = this.faceLandmarker.detectForVideo(videoElement, timestamp);
      const faceAnalysis = this.analyzeFace(faceResults);
      // Analyze posture
      const poseResults = this.poseLandmarker.detectForVideo(videoElement, timestamp);
      const postureAnalysis = this.analyzePosture(poseResults);
      // Analyze gestures
      const gestureResults = this.gestureRecognizer.recognizeForVideo(videoElement, timestamp);
      const gestureAnalysis = this.analyzeGestures(gestureResults);
      // Analyze audio
      const audioAnalysis = this.analyzeAudio();
      // Calculate weighted confidence based on detection quality
      const faceDetected = faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0;
      const poseDetected = poseResults.landmarks && poseResults.landmarks.length > 0;
      const gesturesDetected = gestureResults.gestures && gestureResults.gestures.length > 0;
      const audioDetected = audioAnalysis.volume > 0;

      const detectionQuality = (
        (faceDetected ? 0.35 : 0) +
        (poseDetected ? 0.25 : 0) +
        (gesturesDetected ? 0.15 : 0) +
        (audioDetected ? 0.25 : 0)
      );

      // Enhanced confidence calculation
      const overallConfidence = Math.round(
        (faceAnalysis.eyeContact * 0.25 +
          postureAnalysis.postureScore * 0.25 +
          gestureAnalysis.gestureVariety * 0.15 +
          faceAnalysis.emotionConfidence * 100 * 0.15 +
          audioAnalysis.audioConfidence * 100 * 0.2) * detectionQuality
      );

      return {
        face: {
          ...faceAnalysis,
          landmarks: faceResults?.faceLandmarks?.[0]?.map((l: any) => ({
            x: l.x,
            y: l.y,
            z: l.z || 0
          })) || []
        },
        posture: {
          ...postureAnalysis,
          landmarks: poseResults?.landmarks?.[0]?.map((l: any) => ({
            x: l.x,
            y: l.y,
            z: l.z || 0,
            visibility: l.visibility || 0
          })) || []
        },
        gestures: gestureAnalysis,
        audio: audioAnalysis,
        overallConfidence: Math.max(0, Math.min(100, overallConfidence)),
        timestamp,
      };
    } catch (error) {
      console.error('Error analyzing frame:', error);
      return this.getDefaultMetrics();
    }
  }

  private analyzeFace(results: any): FaceAnalysis {
    if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
      return {
        eyeContact: 0,
        emotion: 'neutral',
        emotionConfidence: 0,
        facialMovement: 0,
        gazeDirection: { x: 0, y: 0 },
      };
    }
    const landmarks = results.faceLandmarks[0];
    const blendshapes = results.faceBlendshapes?.[0]?.categories;
    // === ENHANCED EYE CONTACT TRACKING ===
    const leftEAR = this.calculateEAR(landmarks, 'left');
    const rightEAR = this.calculateEAR(landmarks, 'right');
    const avgEAR = (leftEAR + rightEAR) / 2;
   
    // Iris tracking for precise gaze
    const leftIris = landmarks[468] || landmarks[33];
    const rightIris = landmarks[473] || landmarks[263];
    const noseTip = landmarks[1];
   
    // Calculate gaze vector with improved calibration
    const gazeVector = this.calculateGazeVector(leftIris, rightIris, noseTip, landmarks);
   
    // Enhanced eye contact scoring
    const gazeDistance = Math.sqrt(gazeVector.x * gazeVector.x + gazeVector.y * gazeVector.y);
   
    // Improved EAR scoring (0.2-0.4 is normal open eye)
    const earNormalized = Math.max(0, Math.min(1, (avgEAR - 0.15) / 0.25));
    const eyeOpennessScore = earNormalized * 100;
   
    // Improved gaze scoring with exponential falloff
    const gazeScore = Math.max(0, Math.pow(1 - Math.min(gazeDistance, 1), 1.5) * 100);
   
    // Weighted combination favoring gaze direction
    const eyeContact = Math.round((gazeScore * 0.75 + eyeOpennessScore * 0.25));
    // === ENHANCED MICRO-EXPRESSION ANALYSIS ===
    let facialMovement = 0;
    if (this.previousFaceLandmarks) {
      const criticalRegions = {
        eyebrows: [70, 63, 105, 66, 107, 300, 293, 334, 296, 336], // Brow movement
        eyes: [33, 133, 160, 159, 158, 144, 145, 153, 362, 263, 385, 386, 387, 373, 374, 380], // Eye region
        mouth: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78], // Mouth
        nose: [1, 2, 98, 327], // Nose
      };
     
      let totalMovement = 0;
      let pointsChecked = 0;
     
      Object.values(criticalRegions).forEach(indices => {
        indices.forEach(idx => {
          if (landmarks[idx] && this.previousFaceLandmarks[idx]) {
            const dx = landmarks[idx].x - this.previousFaceLandmarks[idx].x;
            const dy = landmarks[idx].y - this.previousFaceLandmarks[idx].y;
            const dz = (landmarks[idx].z || 0) - (this.previousFaceLandmarks[idx].z || 0);
            const movement = Math.sqrt(dx * dx + dy * dy + dz * dz);
           
            // Only count significant movements
            if (movement > this.MOVEMENT_THRESHOLD) {
              totalMovement += movement;
              pointsChecked++;
            }
          }
        });
      });
     
      // Normalize and scale
      facialMovement = pointsChecked > 0
        ? Math.min(100, (totalMovement / pointsChecked) * 1000)
        : 0;
    }
    this.previousFaceLandmarks = landmarks;
    // === ADVANCED EMOTION DETECTION ===
    const emotion = blendshapes
      ? this.detectEmotionWithBlendshapes(landmarks, blendshapes)
      : this.detectEmotionFACS(landmarks);
    return {
      eyeContact,
      emotion: emotion.label,
      emotionConfidence: emotion.confidence,
      facialMovement: Math.round(facialMovement),
      gazeDirection: gazeVector,
      emotionScores: emotion.scores,
    };
  }

  private detectEmotionWithBlendshapes(landmarks: any[], blendshapes: any[]): {
    label: string;
    confidence: number;
    scores: Record<string, number>;
  } {
    // Convert blendshapes array to map
    const blendshapeMap: Record<string, number> = {};
    blendshapes.forEach((bs: any) => {
      blendshapeMap[bs.categoryName] = bs.score;
    });
    // === EMOTION SCORING using Blendshapes + FACS ===
   
    // HAPPY: Smile + cheek raise + eye squint
    const smileLeft = blendshapeMap['mouthSmileLeft'] || 0;
    const smileRight = blendshapeMap['mouthSmileRight'] || 0;
    const cheekSquintLeft = blendshapeMap['cheekSquintLeft'] || 0;
    const cheekSquintRight = blendshapeMap['cheekSquintRight'] || 0;
    const eyeSquintLeft = blendshapeMap['eyeSquintLeft'] || 0;
    const eyeSquintRight = blendshapeMap['eyeSquintRight'] || 0;
   
    const happyScore = (
      (smileLeft + smileRight) / 2 +
      (cheekSquintLeft + cheekSquintRight) / 2 * 0.6 +
      (eyeSquintLeft + eyeSquintRight) / 2 * 0.4
    ) * 100;
   
    // SAD: Frown + inner brow raise + lip corner down
    const frownLeft = blendshapeMap['mouthFrownLeft'] || 0;
    const frownRight = blendshapeMap['mouthFrownRight'] || 0;
    const browInnerUp = blendshapeMap['browInnerUp'] || 0;
    const mouthShrugUpper = blendshapeMap['mouthShrugUpper'] || 0;
   
    const sadScore = (
      (frownLeft + frownRight) / 2 +
      browInnerUp * 0.8 +
      mouthShrugUpper * 0.4
    ) * 100;
   
    // SURPRISED: Wide eyes + jaw open + brow raise
    const eyeWideLeft = blendshapeMap['eyeWideLeft'] || 0;
    const eyeWideRight = blendshapeMap['eyeWideRight'] || 0;
    const jawOpen = blendshapeMap['jawOpen'] || 0;
    const browOuterUpLeft = blendshapeMap['browOuterUpLeft'] || 0;
    const browOuterUpRight = blendshapeMap['browOuterUpRight'] || 0;
   
    const surprisedScore = (
      (eyeWideLeft + eyeWideRight) / 2 +
      jawOpen * 0.7 +
      (browOuterUpLeft + browOuterUpRight + browInnerUp) / 3 * 0.6
    ) * 100;
   
    // ANGRY: Brow down + eye squint + lip press
    const browDownLeft = blendshapeMap['browDownLeft'] || 0;
    const browDownRight = blendshapeMap['browDownRight'] || 0;
    const noseSneerLeft = blendshapeMap['noseSneerLeft'] || 0;
    const noseSneerRight = blendshapeMap['noseSneerRight'] || 0;
    const mouthPressLeft = blendshapeMap['mouthPressLeft'] || 0;
    const mouthPressRight = blendshapeMap['mouthPressRight'] || 0;
   
    const angryScore = (
      (browDownLeft + browDownRight) / 2 +
      (noseSneerLeft + noseSneerRight) / 2 * 0.6 +
      (mouthPressLeft + mouthPressRight) / 2 * 0.6
    ) * 100;
   
    // FEAR: Wide eyes + brow raise + lip stretch
    const mouthStretchLeft = blendshapeMap['mouthStretchLeft'] || 0;
    const mouthStretchRight = blendshapeMap['mouthStretchRight'] || 0;
    const mouthUpperUpLeft = blendshapeMap['mouthUpperUpLeft'] || 0;
    const mouthUpperUpRight = blendshapeMap['mouthUpperUpRight'] || 0;
   
    const fearScore = (
      (eyeWideLeft + eyeWideRight) / 2 +
      (browInnerUp + browOuterUpLeft + browOuterUpRight) / 3 * 0.7 +
      (mouthStretchLeft + mouthStretchRight) / 2 * 0.3 +
      (mouthUpperUpLeft + mouthUpperUpRight) / 2 * 0.3
    ) * 100;
   
    // DISGUST: Nose wrinkle + upper lip raise + eye squint
    const disgustScore = (
      (noseSneerLeft + noseSneerRight) / 2 +
      (mouthUpperUpLeft + mouthUpperUpRight) / 2 * 0.6 +
      (eyeSquintLeft + eyeSquintRight) / 2 * 0.4
    ) * 100;
   
    // CONTEMPT: Asymmetric smile
    const smileAsymmetry = Math.abs(smileLeft - smileRight);
    const contemptScore = smileAsymmetry * 80 + (noseSneerLeft + noseSneerRight) / 2 * 40;
   
    // NEUTRAL: Low activation
    const totalActivation = happyScore + sadScore + surprisedScore + angryScore + fearScore + disgustScore + contemptScore;
    const neutralScore = Math.max(0, 100 - totalActivation * 0.8);

    // Combine with geometric FACS for validation
    const facsEmotion = this.detectEmotionFACS(landmarks);
   
    // Create emotion map with FACS boost
    const emotions = {
      happy: happyScore * 0.7 + (facsEmotion.label === 'happy' ? facsEmotion.confidence * 30 : 0),
      sad: sadScore * 0.7 + (facsEmotion.label === 'sad' ? facsEmotion.confidence * 30 : 0),
      surprised: surprisedScore * 0.7 + (facsEmotion.label === 'surprised' ? facsEmotion.confidence * 30 : 0),
      angry: angryScore * 0.7 + (facsEmotion.label === 'angry' ? facsEmotion.confidence * 30 : 0),
      fear: fearScore * 0.7 + (facsEmotion.label === 'fear' ? facsEmotion.confidence * 30 : 0),
      disgust: disgustScore * 0.7 + (facsEmotion.label === 'disgust' ? facsEmotion.confidence * 30 : 0),
      contempt: contemptScore,
      neutral: neutralScore,
    };
    // Find dominant emotion
    const sortedEmotions = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
    const dominantEmotion = sortedEmotions[0];
    const secondEmotion = sortedEmotions[1];
   
    // Smooth emotion transitions
    this.emotionHistory.push(dominantEmotion[0]);
    if (this.emotionHistory.length > this.EMOTION_SMOOTHING) {
      this.emotionHistory.shift();
    }
   
    // Use most common emotion in recent history
    const emotionCounts: Record<string, number> = {};
    this.emotionHistory.forEach(e => {
      emotionCounts[e] = (emotionCounts[e] || 0) + 1;
    });
   
    const smoothedEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0][0];
   
    // Calculate confidence
    const totalScore = sortedEmotions.reduce((sum, [_, score]) => sum + Math.max(0, score), 0.001);
    const dominanceRatio = dominantEmotion[1] / totalScore;
    const separationRatio = (dominantEmotion[1] - secondEmotion[1]) / totalScore;
   
    const confidence = Math.min(0.99, dominanceRatio * 0.6 + separationRatio * 0.4);
    return {
      label: smoothedEmotion,
      confidence: Math.max(0.1, confidence),
      scores: emotions,
    };
  }

  private detectEmotionFACS(landmarks: any[]): {
    label: string;
    confidence: number;
    scores?: Record<string, number>;
  } {
    // === OPTIMIZED ACTION UNITS ===
   
    // AU1+2: Brow raise (Surprise, Fear)
    const leftInnerBrow = landmarks[70];
    const rightInnerBrow = landmarks[300];
    const leftOuterBrow = landmarks[107];
    const rightOuterBrow = landmarks[336];
    const browBaseline = landmarks[168];
   
    const browRaise = Math.max(0,
      ((leftInnerBrow.y + rightInnerBrow.y) / 2 - browBaseline.y) * -10 // y decreases upward
    );
   
    // AU4: Brow lower (Anger, Concern)
    const browLower = Math.max(0,
      browBaseline.y - (leftInnerBrow.y + rightInnerBrow.y) / 2
    ) * 10;
   
    // AU6: Cheek raise (Genuine smile)
    const leftCheek = landmarks[205];
    const rightCheek = landmarks[425];
    const leftEyeBottom = landmarks[145];
    const rightEyeBottom = landmarks[374];
    const cheekRaise = Math.max(0,
      ((leftEyeBottom.y - leftCheek.y) + (rightEyeBottom.y - rightCheek.y)) / 2 * 10
    );
   
    // AU12: Lip corner pull (Smile)
    const leftMouthCorner = landmarks[61];
    const rightMouthCorner = landmarks[291];
    const mouthCenter = landmarks[13];
    const smileWidth = this.euclideanDist(leftMouthCorner, rightMouthCorner);
    const smileLift = Math.max(0,
      ((leftMouthCorner.y + rightMouthCorner.y) / 2 - mouthCenter.y) * -16
    );
   
    // AU15: Lip corner depress (Sadness)
    const lipDepress = Math.max(0, -smileLift);
   
    // AU25+26: Jaw drop (Surprise)
    const upperLip = landmarks[13];
    const lowerLip = landmarks[14];
    const mouthOpen = this.euclideanDist(upperLip, lowerLip) * 20;
   
    // AU43: Eye closure
    const leftEyeOpen = this.euclideanDist(landmarks[159], landmarks[145]);
    const rightEyeOpen = this.euclideanDist(landmarks[386], landmarks[374]);
    const eyesOpen = (leftEyeOpen + rightEyeOpen) / 2 * 40;
   
    // === CALIBRATED EMOTION SCORING ===
   
    const emotions = {
      happy: Math.max(0, cheekRaise * 3 + smileLift * 5 + smileWidth * 2),
      sad: Math.max(0, browRaise * 2 + lipDepress * 4 + (20 - eyesOpen) * 0.5),
      surprised: Math.max(0, browRaise * 3 + mouthOpen * 4 + eyesOpen * 1.5),
      angry: Math.max(0, browLower * 5 + (20 - eyesOpen) * 1),
      fear: Math.max(0, browRaise * 2 + eyesOpen * 2 + mouthOpen * 1),
      disgust: Math.max(0, browLower * 2 + lipDepress * 2),
      neutral: 50,
    };
   
    // Adjust neutral
    const emotionSum = Object.values(emotions).reduce((sum, val) => sum + val, 0) - emotions.neutral;
    emotions.neutral = Math.max(0, 50 - emotionSum * 0.5);
   
    const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
    const total = sorted.reduce((sum, [_, s]) => sum + s, 0.001);
    const conf = sorted[0][1] / total;
   
    return {
      label: sorted[0][0],
      confidence: Math.min(0.99, conf),
      scores: emotions,
    };
  }

  private calculateEAR(landmarks: any[], eye: 'left' | 'right'): number {
    let p1, p2, p3, p4, p5, p6;
   
    if (eye === 'left') {
      p1 = landmarks[33];
      p2 = landmarks[160];
      p3 = landmarks[158];
      p4 = landmarks[133];
      p5 = landmarks[153];
      p6 = landmarks[144];
    } else {
      p1 = landmarks[362];
      p2 = landmarks[385];
      p3 = landmarks[387];
      p4 = landmarks[263];
      p5 = landmarks[373];
      p6 = landmarks[380];
    }
   
    const euclidean = (a: any, b: any) => Math.sqrt(
      (a.x - b.x)**2 + (a.y - b.y)**2
    );
   
    const vertical1 = euclidean(p2, p6);
    const vertical2 = euclidean(p3, p5);
    const horizontal = euclidean(p1, p4);
   
    return (vertical1 + vertical2) / (2 * horizontal);
  }

  private calculateGazeVector(
    leftIris: any,
    rightIris: any,
    noseTip: any,
    landmarks: any[]
  ): { x: number; y: number } {
    const leftEyeLeft = landmarks[33];
    const leftEyeRight = landmarks[133];
    const rightEyeLeft = landmarks[362];
    const rightEyeRight = landmarks[263];
   
    const leftEyeWidth = Math.abs(leftEyeRight.x - leftEyeLeft.x);
    const rightEyeWidth = Math.abs(rightEyeRight.x - rightEyeLeft.x);
   
    const leftIrisX = leftEyeWidth > 0
      ? (leftIris.x - leftEyeLeft.x) / leftEyeWidth
      : 0.5;
    const rightIrisX = rightEyeWidth > 0
      ? (rightIris.x - rightEyeLeft.x) / rightEyeWidth
      : 0.5;
   
    const gazeX = ((leftIrisX + rightIrisX) / 2 - 0.5) * 2.2;
    const gazeY = ((leftIris.y + rightIris.y) / 2 - 0.45) * 2.2;
   
    return { x: Math.max(-1, Math.min(1, gazeX)), y: Math.max(-1, Math.min(1, gazeY)) };
  }

  private euclideanDist(p1: any, p2: any): number {
    return Math.sqrt(
      (p1.x - p2.x)**2 +
      (p1.y - p2.y)**2 +
      ((p1.z || 0) - (p2.z || 0))**2
    );
  }

  private analyzePosture(results: any): PostureAnalysis {
    if (!results.landmarks || results.landmarks.length === 0) {
      return {
        postureScore: 0,
        shoulderAlignment: 0,
        headPosition: 0,
        stability: 0,
      };
    }
    const landmarks = results.landmarks[0];
    // Key pose landmarks
    const nose = landmarks[0];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];
    // === IMPROVED POSTURE CALCULATIONS ===
   
    // 1. Shoulder alignment (horizontal level check)
    const shoulderAngle = Math.abs(
      Math.atan2(
        rightShoulder.y - leftShoulder.y,
        rightShoulder.x - leftShoulder.x
      ) * (180 / Math.PI)
    );
    const shoulderAlignment = Math.max(0, Math.min(100, 100 - shoulderAngle * 3));
   
    // 2. Head position (should be centered over shoulders)
    const shoulderMid = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2,
      z: ((leftShoulder.z || 0) + (rightShoulder.z || 0)) / 2,
    };
   
    const headOffset = Math.abs(nose.x - shoulderMid.x);
    const headPosition = Math.max(0, Math.min(100, 100 - headOffset * 150));
   
    // 3. Spine alignment (vertical check)
    const hipMid = {
      x: (leftHip.x + rightHip.x) / 2,
      y: (leftHip.y + rightHip.y) / 2,
      z: ((leftHip.z || 0) + (rightHip.z || 0)) / 2,
    };
    const spineDeviation = Math.abs(shoulderMid.x - hipMid.x);
    const spineAlignment = Math.max(0, Math.min(100, 100 - spineDeviation * 200));
   
    // 4. Neck angle (head tilt)
    const neckPoint = { x: nose.x, y: nose.y - 0.1, z: nose.z }; // Virtual point above head
    const neckAngle = this.calculateAngle(neckPoint, nose, shoulderMid);
    const headUpright = Math.max(0, Math.min(100, 100 - Math.abs(180 - neckAngle) * 1.5));
   
    // Overall posture score
    const postureScore = Math.round(
      shoulderAlignment * 0.3 +
      headPosition * 0.2 +
      spineAlignment * 0.3 +
      headUpright * 0.2
    );
    // === Optical Flow - Movement pattern tracking ===
    let stability = 100;
    if (this.previousPoseLandmarks) {
      let totalMovement = 0;
      const keyPoints = [0, 11, 12, 23, 24]; // Nose, shoulders, hips
     
      for (const idx of keyPoints) {
        if (landmarks[idx] && this.previousPoseLandmarks[idx]) {
          const dx = landmarks[idx].x - this.previousPoseLandmarks[idx].x;
          const dy = landmarks[idx].y - this.previousPoseLandmarks[idx].y;
          const dz = (landmarks[idx].z || 0) - (this.previousPoseLandmarks[idx].z || 0);
          totalMovement += Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
     
      stability = Math.max(0, Math.round(100 - totalMovement * 500));
    }
    this.previousPoseLandmarks = landmarks;
    return {
      postureScore: Math.max(0, postureScore),
      shoulderAlignment: Math.max(0, Math.round(shoulderAlignment)),
      headPosition: Math.max(0, Math.round(headPosition)),
      stability: Math.max(0, stability),
    };
  }

  // Calculate angle between three 3D points (in degrees)
  private calculateAngle(p1: any, p2: any, p3: any): number {
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y, z: (p1.z || 0) - (p2.z || 0) };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y, z: (p3.z || 0) - (p2.z || 0) };
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x**2 + v1.y**2 + v1.z**2);
    const mag2 = Math.sqrt(v2.x**2 + v2.y**2 + v2.z**2);
    const cos = dot / (mag1 * mag2 || 1);
    return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
  }

  private analyzeGestures(results: any): GestureAnalysis {
    const gestures = results.gestures || [];
    const handedness = results.handedness || [];
    // Track gesture variety
    if (gestures.length > 0) {
      for (const gesture of gestures) {
        if (gesture[0]) {
          this.gestureHistory.push(gesture[0].categoryName);
        }
      }
    }
    if (this.gestureHistory.length > 30) {
      this.gestureHistory = this.gestureHistory.slice(-30);
    }
    const uniqueGestures = new Set(this.gestureHistory);
    const gestureVariety = Math.min(100, uniqueGestures.size * 20);
    // Hand visibility
    const handVisibility = Math.min(100, handedness.length * 50);
    // Movement patterns
    const movementPatterns = Array.from(uniqueGestures);
    return {
      gestureCount: gestures.length,
      gestureVariety,
      handVisibility,
      movementPatterns,
    };
  }

  private analyzeAudio(): AudioAnalysis {
    if (!this.analyser || !this.audioContext) {
      return this.getDefaultAudio();
    }

    const bufferLength = this.analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);
    this.analyser.getFloatTimeDomainData(dataArray);

    // Compute energy (RMS)
    let sumSquares = 0;
    for (let i = 0; i < bufferLength; i++) {
      sumSquares += dataArray[i] ** 2;
    }
    const rms = Math.sqrt(sumSquares / bufferLength);

    // Compute zero-crossing rate (ZCR)
    let zeroCrossings = 0;
    for (let i = 1; i < bufferLength; i++) {
      if (dataArray[i - 1] * dataArray[i] < 0) {
        zeroCrossings++;
      }
    }
    const zcr = zeroCrossings / (bufferLength - 1);

    // Voice Activity Detection (VAD)
    const isSpeech = rms > this.ENERGY_THRESHOLD && zcr > this.ZCR_MIN && zcr < this.ZCR_MAX;

    if (!isSpeech) {
      return this.getDefaultAudio();
    }

    // If speech detected, proceed with analysis
    const volume = Math.min(100, rms * 100); // Reduced scaling for less sensitivity

    // Pitch using YIN
    const pitch = this.yinPitchDetection(dataArray, this.audioContext.sampleRate);

    // Voice stability
    this.pitchHistory.push(pitch);
    if (this.pitchHistory.length > 10) {
      this.pitchHistory.shift();
    }
    const pitchMean = this.pitchHistory.reduce((a, b) => a + b, 0) / this.pitchHistory.length || 1;
    const pitchVar = Math.sqrt(
      this.pitchHistory.reduce((a, b) => a + (b - pitchMean) ** 2, 0) / this.pitchHistory.length
    );
    const voiceStability = Math.max(0, Math.min(100, 100 - pitchVar * 2));

    // Speech rate (zero-crossing approximation)
    const speechRate = (zeroCrossings / bufferLength) * (this.audioContext.sampleRate / 2) / 10;

    // Basic audio emotion
    let audioEmotion = 'neutral';
    let audioConfidence = 0.5;
    if (volume > 50 && pitch > 180) {
      audioEmotion = 'excited';
      audioConfidence = 0.8;
    } else if (volume < 25 && pitch < 140) {
      audioEmotion = 'calm';
      audioConfidence = 0.7;
    } else if (pitchVar > 25) {
      audioEmotion = 'nervous';
      audioConfidence = 0.6;
    } else if (volume > 40 && pitchVar < 15) {
      audioEmotion = 'confident';
      audioConfidence = 0.75;
    }

    return {
      pitch: Math.round(pitch || 0),
      volume: Math.round(volume),
      speechRate: Math.round(speechRate),
      voiceStability: Math.round(voiceStability),
      audioEmotion,
      audioConfidence,
    };
  }

  // YIN Pitch Detection Algorithm (optimized for real-time)
  private yinPitchDetection(buffer: Float32Array, sampleRate: number): number {
    const bufferLength = buffer.length / 2;
    const yinBuffer = new Float32Array(bufferLength);
    let runningSum = 0;

    yinBuffer[0] = 1;
    for (let tau = 1; tau < bufferLength; tau++) {
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const diff = buffer[i] - buffer[i + tau];
        sum += diff ** 2;
      }
      yinBuffer[tau] = sum;
      runningSum += sum;
      if (runningSum > 0) yinBuffer[tau] *= tau / runningSum;
    }

    let minTau = -1;
    for (let tau = 2; tau < bufferLength; tau++) {
      if (yinBuffer[tau] < this.YIN_THRESHOLD) {
        minTau = tau;
        break;
      }
    }

    if (minTau === -1) return 0;

    const betterTau = minTau + (yinBuffer[minTau - 1] - yinBuffer[minTau + 1]) / (2 * (yinBuffer[minTau - 1] + yinBuffer[minTau + 1] - 2 * yinBuffer[minTau]));
    return sampleRate / betterTau;
  }

  private getDefaultMetrics(): BodyLanguageMetrics {
    return {
      face: {
        eyeContact: 0,
        emotion: 'neutral',
        emotionConfidence: 0,
        facialMovement: 0,
        gazeDirection: { x: 0, y: 0 },
        landmarks: [],
      },
      posture: {
        postureScore: 0,
        shoulderAlignment: 0,
        headPosition: 0,
        stability: 0,
        landmarks: [],
      },
      gestures: {
        gestureCount: 0,
        gestureVariety: 0,
        handVisibility: 0,
        movementPatterns: [],
      },
      audio: this.getDefaultAudio(),
      overallConfidence: 0,
      timestamp: Date.now(),
    };
  }

  private getDefaultAudio(): AudioAnalysis {
    return {
      pitch: 0,
      volume: 0,
      speechRate: 0,
      voiceStability: 0,
      audioEmotion: 'silent',
      audioConfidence: 0,
    };
  }

  cleanup() {
    if (this.faceLandmarker) {
      this.faceLandmarker.close();
      this.faceLandmarker = null;
    }
    if (this.poseLandmarker) {
      this.poseLandmarker.close();
      this.poseLandmarker = null;
    }
    if (this.gestureRecognizer) {
      this.gestureRecognizer.close();
      this.gestureRecognizer = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.isInitialized = false;
    this.previousFaceLandmarks = null;
    this.previousPoseLandmarks = null;
    this.gestureHistory = [];
    this.emotionHistory = [];
    this.pitchHistory = [];
  }
}
```
