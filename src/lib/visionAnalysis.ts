// Advanced computer vision using MediaPipe (OpenCV + DeepFace equivalent)
import { FaceLandmarker, PoseLandmarker, GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';

export interface FaceAnalysis {
  eyeContact: number; // 0-100
  emotion: string;
  emotionConfidence: number;
  facialMovement: number; // Micro-expressions
  gazeDirection: { x: number; y: number };
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
  private previousFaceLandmarks: any = null;
  private previousPoseLandmarks: any = null;
  private gestureHistory: string[] = [];

  async initialize() {
    if (this.isInitialized) return;

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      // Initialize Face Landmarker for facial analysis
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
      });

      // Initialize Pose Landmarker for posture analysis
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });

      // Initialize Gesture Recognizer for hand gestures
      this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });

      this.isInitialized = true;
      console.log('MediaPipe vision models initialized successfully');
    } catch (error) {
      console.error('Failed to initialize MediaPipe models:', error);
      throw error;
    }
  }

  async analyzeFrame(videoElement: HTMLVideoElement, timestamp: number): Promise<BodyLanguageMetrics> {
    if (!this.isInitialized || !this.faceLandmarker || !this.poseLandmarker || !this.gestureRecognizer) {
      return this.getDefaultMetrics();
    }

    try {
      // Analyze face
      const faceResults = this.faceLandmarker.detectForVideo(videoElement, timestamp);
      const faceAnalysis = this.analyzeFace(faceResults);

      // Analyze posture
      const poseResults = this.poseLandmarker.detectForVideo(videoElement, timestamp);
      const postureAnalysis = this.analyzePosture(poseResults);

      // Analyze gestures
      const gestureResults = this.gestureRecognizer.recognizeForVideo(videoElement, timestamp);
      const gestureAnalysis = this.analyzeGestures(gestureResults);

      // Calculate overall confidence (weighted by detection quality)
      const faceDetected = faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0;
      const poseDetected = poseResults.landmarks && poseResults.landmarks.length > 0;
      const gesturesDetected = gestureResults.gestures && gestureResults.gestures.length > 0;
      
      const detectionQuality = (
        (faceDetected ? 0.4 : 0) +
        (poseDetected ? 0.4 : 0) +
        (gesturesDetected ? 0.2 : 0)
      ) * 100;
      
      const overallConfidence = Math.round(
        (faceAnalysis.eyeContact * 0.3 +
          postureAnalysis.postureScore * 0.4 +
          gestureAnalysis.gestureVariety * 0.3) * (detectionQuality / 100)
      );

      return {
        face: { 
          ...faceAnalysis, 
          landmarks: faceResults?.faceLandmarks?.[0]?.map((l: any) => ({ x: l.x, y: l.y, z: l.z })) || []
        },
        posture: { 
          ...postureAnalysis, 
          landmarks: poseResults?.landmarks?.[0]?.map((l: any) => ({ x: l.x, y: l.y, z: l.z })) || []
        },
        gestures: gestureAnalysis,
        overallConfidence: Math.max(0, overallConfidence),
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

    // === EYE CONTACT TRACKING (MediaPipe Iris + EAR + Gaze Vector) ===
    
    // Eye Aspect Ratio (EAR) for blink detection and eye openness
    const leftEAR = this.calculateEAR(landmarks, 'left');
    const rightEAR = this.calculateEAR(landmarks, 'right');
    const avgEAR = (leftEAR + rightEAR) / 2;
    
    // Iris landmarks for precise gaze tracking (MediaPipe provides iris landmarks)
    const leftIris = landmarks[468] || landmarks[33]; // Left iris center or fallback to eye center
    const rightIris = landmarks[473] || landmarks[263]; // Right iris center or fallback to eye center
    const noseTip = landmarks[1];
    
    // Calculate gaze vector (direction estimation)
    const gazeVector = this.calculateGazeVector(leftIris, rightIris, noseTip, landmarks);
    
    // Eye contact score based on gaze direction and eye openness
    const gazeDistance = Math.sqrt(gazeVector.x * gazeVector.x + gazeVector.y * gazeVector.y);
    const eyeOpennessScore = Math.min(100, avgEAR * 300); // EAR typically 0.2-0.4 when open
    const gazeScore = Math.max(0, (1 - Math.min(gazeDistance * 1.5, 1)) * 100);
    const eyeContact = Math.round((gazeScore * 0.7 + eyeOpennessScore * 0.3));

    // === MICRO-EXPRESSION ANALYSIS (Temporal frame-to-frame comparison) ===
    let facialMovement = 0;
    if (this.previousFaceLandmarks) {
      // Lucas-Kanade Optical Flow approximation using landmark displacement
      let totalMovement = 0;
      const criticalPoints = [
        ...Array.from({length: 17}, (_, i) => i), // Jawline
        ...Array.from({length: 10}, (_, i) => 17 + i), // Eyebrows
        ...Array.from({length: 20}, (_, i) => 48 + i), // Mouth
      ];
      
      for (const idx of criticalPoints) {
        if (landmarks[idx] && this.previousFaceLandmarks[idx]) {
          const dx = landmarks[idx].x - this.previousFaceLandmarks[idx].x;
          const dy = landmarks[idx].y - this.previousFaceLandmarks[idx].y;
          const dz = (landmarks[idx].z || 0) - (this.previousFaceLandmarks[idx].z || 0);
          totalMovement += Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
      facialMovement = Math.min(100, totalMovement * 500);
    }
    this.previousFaceLandmarks = landmarks;

    // === FACS-based EMOTION DETECTION ===
    const emotion = this.detectEmotionFACS(landmarks);

    return {
      eyeContact,
      emotion: emotion.label,
      emotionConfidence: emotion.confidence,
      facialMovement: Math.round(facialMovement),
      gazeDirection: gazeVector,
    };
  }

  // Calculate Eye Aspect Ratio (EAR) for blink detection
  private calculateEAR(landmarks: any[], eye: 'left' | 'right'): number {
    // EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    // Vertical eye landmarks / horizontal eye landmarks
    
    let p1, p2, p3, p4, p5, p6;
    
    if (eye === 'left') {
      // Left eye landmarks (MediaPipe Face Mesh indices)
      p1 = landmarks[33];  // Left corner
      p2 = landmarks[160]; // Top left
      p3 = landmarks[158]; // Top right  
      p4 = landmarks[133]; // Right corner
      p5 = landmarks[153]; // Bottom right
      p6 = landmarks[144]; // Bottom left
    } else {
      // Right eye landmarks
      p1 = landmarks[362]; // Left corner
      p2 = landmarks[385]; // Top left
      p3 = landmarks[387]; // Top right
      p4 = landmarks[263]; // Right corner
      p5 = landmarks[373]; // Bottom right
      p6 = landmarks[380]; // Bottom left
    }
    
    const euclidean = (a: any, b: any) => Math.sqrt(
      Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2)
    );
    
    const vertical1 = euclidean(p2, p6);
    const vertical2 = euclidean(p3, p5);
    const horizontal = euclidean(p1, p4);
    
    return (vertical1 + vertical2) / (2 * horizontal);
  }

  // Calculate gaze vector for direction estimation
  private calculateGazeVector(
    leftIris: any, 
    rightIris: any, 
    noseTip: any,
    landmarks: any[]
  ): { x: number; y: number } {
    // Calculate iris center relative to eye corners
    const leftEyeLeft = landmarks[33];
    const leftEyeRight = landmarks[133];
    const rightEyeLeft = landmarks[362];
    const rightEyeRight = landmarks[263];
    
    // Normalized iris position within eye (0.5 = center, <0.5 = left, >0.5 = right)
    const leftIrisX = (leftIris.x - leftEyeLeft.x) / (leftEyeRight.x - leftEyeLeft.x);
    const rightIrisX = (rightIris.x - rightEyeLeft.x) / (rightEyeRight.x - rightEyeLeft.x);
    
    // Average gaze direction
    const gazeX = ((leftIrisX + rightIrisX) / 2 - 0.5) * 2; // Normalize to -1 to 1
    const gazeY = ((leftIris.y + rightIris.y) / 2 - 0.5) * 2;
    
    return { x: gazeX, y: gazeY };
  }

  // FACS (Facial Action Coding System) - Maps muscle movements to emotions
  // Uses Euclidean distance ratios between landmarks
  private detectEmotionFACS(landmarks: any[]): { label: string; confidence: number } {
    // === Action Units (AU) Detection based on FACS ===
    
    // AU1+2: Inner/Outer Brow Raiser (Surprise, Fear)
    const leftInnerBrow = landmarks[70];
    const rightInnerBrow = landmarks[300];
    const leftOuterBrow = landmarks[107];
    const rightOuterBrow = landmarks[336];
    const browRaise = this.euclideanDist(leftInnerBrow, rightInnerBrow) * 2;
    
    // AU4: Brow Lowerer (Anger, Concern)
    const browBaseline = landmarks[168]; // Between eyes
    const browLower = (leftInnerBrow.y + rightInnerBrow.y) / 2 - browBaseline.y;
    
    // AU6: Cheek Raiser (Genuine smile)
    const leftCheek = landmarks[205];
    const rightCheek = landmarks[425];
    const leftEyeBottom = landmarks[145];
    const rightEyeBottom = landmarks[374];
    const cheekRaise = (
      (leftCheek.y - leftEyeBottom.y) + 
      (rightCheek.y - rightEyeBottom.y)
    ) / 2;
    
    // AU9: Nose Wrinkler (Disgust)
    const noseLeft = landmarks[98];
    const noseRight = landmarks[327];
    const noseTip = landmarks[1];
    const noseWrinkle = this.euclideanDist(noseLeft, noseRight) / this.euclideanDist(noseLeft, noseTip);
    
    // AU10: Upper Lip Raiser (Disgust)
    const upperLipTop = landmarks[0];
    const upperLipBottom = landmarks[13];
    const lipRaise = upperLipTop.y - upperLipBottom.y;
    
    // AU12: Lip Corner Puller (Smile)
    const leftMouthCorner = landmarks[61];
    const rightMouthCorner = landmarks[291];
    const mouthCenter = landmarks[13];
    const smileWidth = this.euclideanDist(leftMouthCorner, rightMouthCorner);
    const smileLift = mouthCenter.y - ((leftMouthCorner.y + rightMouthCorner.y) / 2);
    
    // AU15: Lip Corner Depressor (Sadness)
    const lipDepress = -smileLift;
    
    // AU20: Lip Stretcher (Fear)
    const lipStretch = smileWidth;
    
    // AU25: Lips Part (Surprise, Concentration)
    const upperLip = landmarks[13];
    const lowerLip = landmarks[14];
    const mouthOpen = this.euclideanDist(upperLip, lowerLip);
    
    // AU26: Jaw Drop (Surprise)
    const jawDrop = mouthOpen * 2;
    
    // AU43: Eyes Closed (Blink, Negative emotion)
    const leftEyeOpen = this.euclideanDist(landmarks[159], landmarks[145]);
    const rightEyeOpen = this.euclideanDist(landmarks[386], landmarks[374]);
    const eyesOpen = (leftEyeOpen + rightEyeOpen) / 2;
    
    // === Emotion Classification based on AU combinations ===
    
    // Happy: AU6 + AU12 (Cheek raise + Smile)
    const happyScore = (cheekRaise * 10 + smileLift * 50) * (smileWidth > 0.15 ? 1.5 : 1);
    
    // Sad: AU1 + AU4 + AU15 (Inner brow raise + brow lower + lip corners down)
    const sadScore = (browRaise * 20 + Math.abs(browLower) * 30 + lipDepress * 40);
    
    // Surprised: AU1 + AU2 + AU5 + AU26 (Brows raised + eyes wide + jaw drop)
    const surprisedScore = (browRaise * 30 + eyesOpen * 50 + jawDrop * 20);
    
    // Angry: AU4 + AU7 + AU23 (Brow lower + lid tightener + lip tightener)
    const angryScore = Math.abs(browLower) * 50 + (eyesOpen < 0.02 ? 20 : 0);
    
    // Fear: AU1 + AU2 + AU4 + AU5 + AU20 (Mixed brow + eyes wide + lip stretch)
    const fearScore = (browRaise * 20 + eyesOpen * 30 + lipStretch * 0.5);
    
    // Disgust: AU9 + AU15 + AU16 (Nose wrinkle + lip corner down + lower lip depressor)
    const disgustScore = (noseWrinkle * 100 + lipDepress * 30 + lipRaise * 20);
    
    // Neutral: Low scores on all emotions
    const neutralScore = 50 - (happyScore + sadScore + surprisedScore + angryScore + fearScore + disgustScore);
    
    // Find dominant emotion
    const emotions = [
      { label: 'happy', score: happyScore },
      { label: 'sad', score: sadScore },
      { label: 'surprised', score: surprisedScore },
      { label: 'angry', score: angryScore },
      { label: 'fear', score: fearScore },
      { label: 'disgust', score: disgustScore },
      { label: 'neutral', score: Math.max(0, neutralScore) },
    ];
    
    emotions.sort((a, b) => b.score - a.score);
    
    const totalScore = emotions.reduce((sum, e) => sum + Math.max(0, e.score), 0);
    const confidence = totalScore > 0 ? Math.min(1, emotions[0].score / totalScore) : 0;
    
    return {
      label: emotions[0].label,
      confidence: Math.round(confidence * 100) / 100,
    };
  }
  
  private euclideanDist(p1: any, p2: any): number {
    return Math.sqrt(
      Math.pow(p1.x - p2.x, 2) + 
      Math.pow(p1.y - p2.y, 2) + 
      Math.pow((p1.z || 0) - (p2.z || 0), 2)
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

    // === MediaPipe Pose - 33 body keypoints ===
    const nose = landmarks[0];
    const leftEye = landmarks[2];
    const rightEye = landmarks[5];
    const leftEar = landmarks[7];
    const rightEar = landmarks[8];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftElbow = landmarks[13];
    const rightElbow = landmarks[14];
    const leftWrist = landmarks[15];
    const rightWrist = landmarks[16];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];
    const leftKnee = landmarks[25];
    const rightKnee = landmarks[26];

    // === Angle Calculations (Joint angle measurement) ===
    
    // Shoulder angle (slouch detection)
    const shoulderAngle = this.calculateAngle(leftShoulder, nose, rightShoulder);
    const shoulderAlignment = Math.max(0, Math.min(100, 
      // Ideal shoulder angle is ~180° (straight), penalize deviation
      100 - Math.abs(180 - shoulderAngle) * 2
    ));
    
    // Neck angle (head tilt detection)
    const neckAngle = this.calculateAngle(
      { x: nose.x, y: nose.y - 0.1, z: nose.z }, // Virtual point above head
      nose,
      { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2, z: (leftShoulder.z + rightShoulder.z) / 2 }
    );
    const headUpright = Math.max(0, Math.min(100, 
      // Ideal neck angle is ~180° (straight)
      100 - Math.abs(180 - neckAngle) * 1.5
    ));
    
    // Spine alignment (torso angle)
    const shoulderMid = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2,
      z: ((leftShoulder.z || 0) + (rightShoulder.z || 0)) / 2,
    };
    const hipMid = {
      x: (leftHip.x + rightHip.x) / 2,
      y: (leftHip.y + rightHip.y) / 2,
      z: ((leftHip.z || 0) + (rightHip.z || 0)) / 2,
    };
    
    // Check vertical alignment (x deviation should be minimal)
    const spineDeviation = Math.abs(shoulderMid.x - hipMid.x);
    const spineAlignment = Math.max(0, Math.min(100, 100 - spineDeviation * 200));
    
    // Head position (should be centered over shoulders)
    const headCenteredness = 1 - Math.abs(nose.x - shoulderMid.x) * 2;
    const headPosition = Math.max(0, Math.round(headCenteredness * 100));
    
    // Overall posture score (weighted combination)
    const postureScore = Math.round(
      shoulderAlignment * 0.3 +
      headUpright * 0.25 +
      spineAlignment * 0.25 +
      headPosition * 0.2
    );

    // === Optical Flow - Movement pattern tracking ===
    let stability = 100;
    if (this.previousPoseLandmarks) {
      let totalMovement = 0;
      const keyPoints = [0, 11, 12, 13, 14, 15, 16, 23, 24]; // Nose, shoulders, elbows, wrists, hips
      
      for (const idx of keyPoints) {
        if (landmarks[idx] && this.previousPoseLandmarks[idx]) {
          const dx = landmarks[idx].x - this.previousPoseLandmarks[idx].x;
          const dy = landmarks[idx].y - this.previousPoseLandmarks[idx].y;
          const dz = (landmarks[idx].z || 0) - (this.previousPoseLandmarks[idx].z || 0);
          totalMovement += Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
      
      // Normalize movement (lower = more stable)
      // Excessive movement indicates poor stability
      stability = Math.max(0, Math.round((1 - Math.min(totalMovement * 20, 1)) * 100));
    }
    this.previousPoseLandmarks = landmarks;

    return {
      postureScore: Math.max(0, postureScore),
      shoulderAlignment: Math.max(0, Math.round(shoulderAlignment)),
      headPosition: Math.max(0, headPosition),
      stability: Math.max(0, stability),
    };
  }
  
  // Calculate angle between three 3D points (in degrees)
  private calculateAngle(p1: any, p2: any, p3: any): number {
    // Vectors from p2 to p1 and p2 to p3
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
    
    // Dot product
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    
    // Magnitudes
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    
    // Angle in radians
    const cosAngle = dot / (mag1 * mag2);
    const angleRad = Math.acos(Math.max(-1, Math.min(1, cosAngle))); // Clamp to avoid NaN
    
    // Convert to degrees
    return (angleRad * 180) / Math.PI;
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
      overallConfidence: 0,
      timestamp: 0,
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
    this.isInitialized = false;
    this.previousFaceLandmarks = null;
    this.previousPoseLandmarks = null;
    this.gestureHistory = [];
  }
}
