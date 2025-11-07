// Advanced computer vision using MediaPipe (OpenCV + DeepFace equivalent)

import { FaceLandmarker, PoseLandmarker, GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';

import { pipeline, env } from '@huggingface/transformers';


// Configure transformers.js

env.allowLocalModels = false;

env.useBrowserCache = true;


let emotionDetector: any = null;


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


      // Initialize Hugging Face emotion detector

      emotionDetector = await pipeline(

        'image-classification',

        'trpakov/vit-face-expression',  // High-accuracy ViT model for emotions

        { device: 'webgpu' }

      );


      this.isInitialized = true;

      console.log('MediaPipe vision models and emotion detector initialized successfully');

    } catch (error) {

      console.error('Failed to initialize models:', error);

      throw error;

    }

  }


  async analyzeFrame(videoElement: HTMLVideoElement, timestamp: number): Promise<BodyLanguageMetrics> {

    if (!this.isInitialized || !this.faceLandmarker || !this.poseLandmarker || !this.gestureRecognizer || !emotionDetector) {

      return this.getDefaultMetrics();

    }


    try {

      // Analyze face

      const faceResults = this.faceLandmarker.detectForVideo(videoElement, timestamp);

      const faceAnalysis = await this.analyzeFace(faceResults, videoElement);


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


  private async analyzeFace(results: any, videoElement: HTMLVideoElement): Promise<FaceAnalysis> {

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


    // === Improved EMOTION DETECTION using Hugging Face ViT Model ===

    const { emotion, confidence } = await this.detectEmotionML(videoElement, landmarks);


    return {

      eyeContact,

      emotion,

      emotionConfidence: confidence,

      facialMovement: Math.round(facialMovement),

      gazeDirection: gazeVector,

    };

  }


  // Helper to crop face for emotion detection

  private cropFace(videoElement: HTMLVideoElement, landmarks: any[]): string | null {

    const canvas = document.createElement('canvas');

    canvas.width = videoElement.videoWidth;

    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');

    if (!ctx) return null;


    ctx.drawImage(videoElement, 0, 0);


    // Compute bounding box from landmarks (normalized 0-1)

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    landmarks.forEach(landmark => {

      minX = Math.min(minX, landmark.x);

      minY = Math.min(minY, landmark.y);

      maxX = Math.max(maxX, landmark.x);

      maxY = Math.max(maxY, landmark.y);

    });


    // Add padding (10% of bbox size)

    const padding = 0.1;

    const width = maxX - minX;

    const height = maxY - minY;

    minX = Math.max(0, minX - padding * width);

    minY = Math.max(0, minY - padding * height);

    maxX = Math.min(1, maxX + padding * width);

    maxY = Math.min(1, maxY + padding * height);


    // Convert to pixel coordinates

    const pxMinX = minX * canvas.width;

    const pxMinY = minY * canvas.height;

    const pxWidth = (maxX - minX) * canvas.width;

    const pxHeight = (maxY - minY) * canvas.height;


    // Create new canvas for crop

    const cropCanvas = document.createElement('canvas');

    cropCanvas.width = pxWidth;

    cropCanvas.height = pxHeight;

    const cropCtx = cropCanvas.getContext('2d');

    if (!cropCtx) return null;


    cropCtx.drawImage(

      canvas,

      pxMinX, pxMinY, pxWidth, pxHeight,  // Source rect

      0, 0, pxWidth, pxHeight              // Dest rect

    );


    return cropCanvas.toDataURL('image/jpeg', 0.8);

  }


  // ML-based emotion detection using ViT

  private async detectEmotionML(videoElement: HTMLVideoElement, landmarks: any[]): Promise<{ emotion: string; confidence: number }> {

    try {

      const croppedImage = this.cropFace(videoElement, landmarks);

      if (!croppedImage) return { emotion: 'neutral', confidence: 0 };


      const results = await emotionDetector(croppedImage);

      const sorted = results.sort((a: any, b: any) => b.score - a.score);

      return {

        emotion: sorted[0].label,

        confidence: sorted[0].score,

      };

    } catch (error) {

      console.error('Error in ML emotion detection:', error);

      return { emotion: 'neutral', confidence: 0 };

    }

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

      timestamp: Date.now(),

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

