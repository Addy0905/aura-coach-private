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
  face: FaceAnalysis;
  posture: PostureAnalysis;
  gestures: GestureAnalysis;
  overallConfidence: number;
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

      // Calculate overall confidence
      const overallConfidence = Math.round(
        (faceAnalysis.eyeContact * 0.3 +
          postureAnalysis.postureScore * 0.4 +
          gestureAnalysis.gestureVariety * 0.3)
      );

      return {
        face: faceAnalysis,
        posture: postureAnalysis,
        gestures: gestureAnalysis,
        overallConfidence: Math.max(25, overallConfidence),
      };
    } catch (error) {
      console.error('Error analyzing frame:', error);
      return this.getDefaultMetrics();
    }
  }

  private analyzeFace(results: any): FaceAnalysis {
    if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
      return {
        eyeContact: 25,
        emotion: 'neutral',
        emotionConfidence: 0,
        facialMovement: 0,
        gazeDirection: { x: 0, y: 0 },
      };
    }

    const landmarks = results.faceLandmarks[0];

    // Calculate eye contact based on gaze direction
    // Using eye landmarks to estimate gaze
    const leftEye = landmarks[468]; // Left iris
    const rightEye = landmarks[473]; // Right iris
    const noseTip = landmarks[1];

    const gazeX = (leftEye.x + rightEye.x) / 2 - 0.5;
    const gazeY = (leftEye.y + rightEye.y) / 2 - 0.5;

    // Eye contact score: higher when looking at camera (center)
    const gazeDistance = Math.sqrt(gazeX * gazeX + gazeY * gazeY);
    const eyeContact = Math.max(25, Math.round((1 - Math.min(gazeDistance * 2, 1)) * 100));

    // Detect facial movement (micro-expressions) by comparing with previous frame
    let facialMovement = 0;
    if (this.previousFaceLandmarks) {
      let totalMovement = 0;
      for (let i = 0; i < Math.min(landmarks.length, this.previousFaceLandmarks.length); i++) {
        const dx = landmarks[i].x - this.previousFaceLandmarks[i].x;
        const dy = landmarks[i].y - this.previousFaceLandmarks[i].y;
        totalMovement += Math.sqrt(dx * dx + dy * dy);
      }
      facialMovement = Math.min(100, totalMovement * 1000);
    }
    this.previousFaceLandmarks = landmarks;

    // Estimate emotion based on facial landmarks geometry
    const emotion = this.estimateEmotion(landmarks);

    return {
      eyeContact,
      emotion: emotion.label,
      emotionConfidence: emotion.confidence,
      facialMovement: Math.round(facialMovement),
      gazeDirection: { x: gazeX, y: gazeY },
    };
  }

  private estimateEmotion(landmarks: any[]): { label: string; confidence: number } {
    // Simplified emotion detection based on facial geometry
    // In production, you'd use a dedicated emotion detection model
    
    // Get key points
    const leftMouth = landmarks[61];
    const rightMouth = landmarks[291];
    const topLip = landmarks[13];
    const bottomLip = landmarks[14];
    const leftEyebrow = landmarks[70];
    const rightEyebrow = landmarks[300];

    // Calculate mouth openness
    const mouthHeight = Math.abs(topLip.y - bottomLip.y);
    const mouthWidth = Math.abs(leftMouth.x - rightMouth.x);
    const mouthRatio = mouthHeight / mouthWidth;

    // Calculate eyebrow position (raised = surprised/happy, lowered = angry/sad)
    const eyebrowHeight = (leftEyebrow.y + rightEyebrow.y) / 2;

    // Simple emotion classification
    if (mouthRatio > 0.3 && eyebrowHeight < 0.35) {
      return { label: 'surprised', confidence: 0.7 };
    } else if (mouthRatio > 0.15 && mouthWidth > 0.15) {
      return { label: 'happy', confidence: 0.8 };
    } else if (eyebrowHeight < 0.3) {
      return { label: 'concerned', confidence: 0.6 };
    } else {
      return { label: 'neutral', confidence: 0.9 };
    }
  }

  private analyzePosture(results: any): PostureAnalysis {
    if (!results.landmarks || results.landmarks.length === 0) {
      return {
        postureScore: 25,
        shoulderAlignment: 25,
        headPosition: 25,
        stability: 25,
      };
    }

    const landmarks = results.landmarks[0];

    // Key pose landmarks
    const nose = landmarks[0];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];

    // Calculate shoulder alignment
    const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
    const shoulderAlignment = Math.max(25, Math.round((1 - shoulderDiff * 5) * 100));

    // Calculate head position (should be centered and upright)
    const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
    const headCenteredness = 1 - Math.abs(nose.x - shoulderMidX) * 2;
    const headPosition = Math.max(25, Math.round(headCenteredness * 100));

    // Calculate overall posture (spine alignment)
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const hipMidY = (leftHip.y + rightHip.y) / 2;
    const spineAlignment = 1 - Math.abs((nose.y - shoulderMidY) / (shoulderMidY - hipMidY) - 0.5);
    const postureScore = Math.max(25, Math.round(spineAlignment * 100));

    // Calculate stability by comparing with previous frame
    let stability = 75;
    if (this.previousPoseLandmarks) {
      let totalMovement = 0;
      for (let i = 0; i < Math.min(landmarks.length, this.previousPoseLandmarks.length); i++) {
        const dx = landmarks[i].x - this.previousPoseLandmarks[i].x;
        const dy = landmarks[i].y - this.previousPoseLandmarks[i].y;
        totalMovement += Math.sqrt(dx * dx + dy * dy);
      }
      // Lower movement = higher stability
      stability = Math.max(25, Math.round((1 - Math.min(totalMovement * 10, 1)) * 100));
    }
    this.previousPoseLandmarks = landmarks;

    return {
      postureScore,
      shoulderAlignment,
      headPosition,
      stability,
    };
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
        eyeContact: 25,
        emotion: 'neutral',
        emotionConfidence: 0,
        facialMovement: 0,
        gazeDirection: { x: 0, y: 0 },
      },
      posture: {
        postureScore: 25,
        shoulderAlignment: 25,
        headPosition: 25,
        stability: 25,
      },
      gestures: {
        gestureCount: 0,
        gestureVariety: 25,
        handVisibility: 25,
        movementPatterns: [],
      },
      overallConfidence: 25,
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
