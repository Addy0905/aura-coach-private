// src/lib/visionAnalysis.ts
import { FaceLandmarker, PoseLandmarker, GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';

export interface FaceAnalysis {
  eyeContact: number;
  emotion: string;
  emotionConfidence: number;
  facialMovement: number;
  gazeDirection: { x: number; y: number };
  emotionScores?: Record<string, number>;
}

export interface PostureAnalysis {
  postureScore: number;
  shoulderAlignment: number;
  headPosition: number;
  stability: number;
}

export interface GestureAnalysis {
  gestureCount: number;
  gestureVariety: number;
  handVisibility: number;
  movementPatterns: string[];
}

export interface AudioAnalysis {
  pitch: number;
  volume: number;
  speechRate: number;
  voiceStability: number;
  audioEmotion: string;
  audioConfidence: number;
  clarity: number; // ← NEW: 0–100 real clarity
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

  // Audio clarity calibration
  private noiseFloor = -60;
  private noiseCalibrated = false;
  private calibrationSamples: number[] = [];
  private readonly VOICE_THRESHOLD_DB = -45;
  private readonly MIN_PITCH = 80;
  private readonly MAX_PITCH = 500;
  private readonly HISTORY_SIZE = 50;

  // Thresholds
  private readonly EAR_THRESHOLD = 0.21;
  private readonly BLINK_FRAMES = 2;
  private readonly EMOTION_SMOOTHING = 5;
  private readonly MOVEMENT_THRESHOLD = 0.002;
  private readonly SAMPLE_RATE = 44100;
  private readonly FFT_SIZE = 2048;
  private readonly YIN_THRESHOLD = 0.1;

  async initialize(videoElement?: HTMLVideoElement) {
    if (this.isInitialized) return;
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

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
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });

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
      console.log('MediaPipe vision and audio models initialized');
    } catch (error) {
      console.error('Failed to initialize MediaPipe models:', error);
      throw error;
    }
  }

  async analyzeFrame(videoElement: HTMLVideoElement, timestamp: number): Promise<BodyLanguageMetrics> {
    if (!this.isInitialized || !this.faceLandmarker || !this.poseLandmarker || !this.gestureRecognizer) {
      return this.getDefaultMetrics();
    }

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

      const faceResults = this.faceLandmarker.detectForVideo(videoElement, timestamp);
      const faceAnalysis = this.analyzeFace(faceResults);

      const poseResults = this.poseLandmarker.detectForVideo(videoElement, timestamp);
      const postureAnalysis = this.analyzePosture(poseResults);

      const gestureResults = this.gestureRecognizer.recognizeForVideo(videoElement, timestamp);
      const gestureAnalysis = this.analyzeGestures(gestureResults);

      const audioAnalysis = this.analyzeAudio(); // ← Now includes clarity

      const faceDetected = !!faceResults.faceLandmarks?.[0];
      const poseDetected = !!poseResults.landmarks?.[0];
      const gesturesDetected = gestureResults.gestures?.length > 0;
      const audioDetected = audioAnalysis.volume > 0;

      const detectionQuality = (faceDetected ? 0.35 : 0) + (poseDetected ? 0.25 : 0) + (gesturesDetected ? 0.15 : 0) + (audioDetected ? 0.25 : 0);

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
          landmarks: faceResults.faceLandmarks?.[0]?.map((l: any) => ({ x: l.x, y: l.y, z: l.z || 0 })) || []
        },
        posture: {
          ...postureAnalysis,
          landmarks: poseResults.landmarks?.[0]?.map((l: any) => ({ x: l.x, y: l.y, z: l.z || 0, visibility: l.visibility || 0 })) || []
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
    if (!results.faceLandmarks?.[0]) {
      return { eyeContact: 0, emotion: 'neutral', emotionConfidence: 0, facialMovement: 0, gazeDirection: { x: 0, y: 0 } };
    }

    const landmarks = results.faceLandmarks[0];
    const blendshapes = results.faceBlendshapes?.[0]?.categories;

    // Eye contact
    const leftEAR = this.calculateEAR(landmarks, 'left');
    const rightEAR = this.calculateEAR(landmarks, 'right');
    const avgEAR = (leftEAR + rightEAR) / 2;
    const leftIris = landmarks[468] || landmarks[33];
    const rightIris = landmarks[473] || landmarks[263];
    const noseTip = landmarks[1];
    const gazeVector = this.calculateGazeVector(leftIris, rightIris, noseTip, landmarks);
    const gazeDistance = Math.sqrt(gazeVector.x ** 2 + gazeVector.y ** 2);
    const eyeOpennessScore = Math.max(0, Math.min(1, (avgEAR - 0.15) / 0.25)) * 100;
    const gazeScore = Math.max(0, Math.pow(1 - Math.min(gazeDistance, 1), 1.5) * 100);
    const eyeContact = Math.round(gazeScore * 0.75 + eyeOpennessScore * 0.25);

    // Facial movement
    let facialMovement = 0;
    if (this.previousFaceLandmarks) {
      const regions = {
        eyebrows: [70, 63, 105, 66, 107, 300, 293, 334, 296, 336],
        eyes: [33, 133, 160, 159, 158, 144, 145, 153, 362, 263, 385, 386, 387, 373, 374, 380],
        mouth: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
        nose: [1, 2, 98, 327],
      };
      let total = 0, count = 0;
      Object.values(regions).forEach(indices => {
        indices.forEach(i => {
          if (landmarks[i] && this.previousFaceLandmarks[i]) {
            const d = this.euclideanDist(landmarks[i], this.previousFaceLandmarks[i]);
            if (d > this.MOVEMENT_THRESHOLD) { total += d; count++; }
          }
        });
      });
      facialMovement = count > 0 ? Math.min(100, (total / count) * 1000) : 0;
    }
    this.previousFaceLandmarks = landmarks;

    const emotion = blendshapes ? this.detectEmotionWithBlendshapes(landmarks, blendshapes) : this.detectEmotionFACS(landmarks);

    return {
      eyeContact,
      emotion: emotion.label,
      emotionConfidence: emotion.confidence,
      facialMovement: Math.round(facialMovement),
      gazeDirection: gazeVector,
      emotionScores: emotion.scores,
    };
  }

  // ... [detectEmotionWithBlendshapes, detectEmotionFACS, calculateEAR, calculateGazeVector, euclideanDist] unchanged ...

  private analyzePosture(results: any): PostureAnalysis {
    if (!results.landmarks?.[0]) {
      return { postureScore: 0, shoulderAlignment: 0, headPosition: 0, stability: 0 };
    }

    const l = results.landmarks[0];
    const nose = l[0], leftShoulder = l[11], rightShoulder = l[12], leftHip = l[23], rightHip = l[24];

    const shoulderAngle = Math.abs(Math.atan2(rightShoulder.y - leftShoulder.y, rightShoulder.x - leftShoulder.x) * (180 / Math.PI));
    const shoulderAlignment = Math.max(0, Math.min(100, 100 - shoulderAngle * 3));

    const shoulderMid = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
    const headOffset = Math.abs(nose.x - shoulderMid.x);
    const headPosition = Math.max(0, Math.min(100, 100 - headOffset * 150));

    const hipMid = { x: (leftHip.x + rightHip.x) / 2 };
    const spineDeviation = Math.abs(shoulderMid.x - hipMid.x);
    const spineAlignment = Math.max(0, Math.min(100, 100 - spineDeviation * 200));

    const neckAngle = this.calculateAngle({ x: nose.x, y: nose.y - 0.1 }, nose, shoulderMid);
    const headUpright = Math.max(0, Math.min(100, 100 - Math.abs(180 - neckAngle) * 1.5));

    const postureScore = Math.round(shoulderAlignment * 0.3 + headPosition * 0.2 + spineAlignment * 0.3 + headUpright * 0.2);

    let stability = 100;
    if (this.previousPoseLandmarks) {
      let total = 0;
      [0, 11, 12, 23, 24].forEach(i => {
        if (l[i] && this.previousPoseLandmarks[i]) {
          total += this.euclideanDist(l[i], this.previousPoseLandmarks[i]);
        }
      });
      stability = Math.max(0, 100 - total * 500);
    }
    this.previousPoseLandmarks = l;

    return {
      postureScore: Math.max(0, postureScore),
      shoulderAlignment: Math.round(shoulderAlignment),
      headPosition: Math.round(headPosition),
      stability: Math.round(stability),
    };
  }

  private analyzeGestures(results: any): GestureAnalysis {
    const gestures = results.gestures || [];
    gestures.forEach((g: any) => g[0] && this.gestureHistory.push(g[0].categoryName));
    if (this.gestureHistory.length > 30) this.gestureHistory = this.gestureHistory.slice(-30);
    const variety = Math.min(100, new Set(this.gestureHistory).size * 20);
    const visibility = Math.min(100, (results.handedness || []).length * 50);
    return {
      gestureCount: gestures.length,
      gestureVariety: variety,
      handVisibility: visibility,
      movementPatterns: Array.from(new Set(this.gestureHistory)),
    };
  }

  // UPDATED: Clarity starts at 0
  private analyzeAudio(): AudioAnalysis {
    if (!this.analyser) {
      return {
        pitch: 0, volume: 0, speechRate: 0, voiceStability: 0,
        audioEmotion: 'silent', audioConfidence: 0, clarity: 0
      };
    }

    const buffer = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buffer);

    // RMS Volume
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) sum += buffer[i] ** 2;
    const rms = Math.sqrt(sum / buffer.length);
    const volume = Math.min(100, rms * 200);
    const volDB = rms > 0 ? 20 * Math.log10(rms) : -Infinity;

    // Calibrate noise floor
    if (!this.noiseCalibrated) {
      this.calibrationSamples.push(volDB);
      if (this.calibrationSamples.length >= 20) {
        const sorted = [...this.calibrationSamples].sort((a, b) => a - b);
        this.noiseFloor = sorted[Math.floor(sorted.length / 2)];
        this.noiseCalibrated = true;
      }
    }

    const snr = this.noiseCalibrated ? Math.max(0, volDB - this.noiseFloor) : 0;
    const voiceActive = volDB > this.VOICE_THRESHOLD_DB && snr > 0;

    // Pitch (YIN)
    const pitch = voiceActive ? this.yinPitchDetection(buffer, this.audioContext!.sampleRate) : 0;
    if (pitch > 0) {
      this.pitchHistory.push(pitch);
      if (this.pitchHistory.length > this.HISTORY_SIZE) this.pitchHistory.shift();
    }

    // Clarity: SNR + ZCR + Centroid
    let clarity = 0;
    if (voiceActive) {
      const freq = new Uint8Array(this.analyser.frequencyBinCount);
      this.analyser.getByteFrequencyData(freq);

      const centroid = this.spectralCentroid(freq);
      const zcr = this.zeroCrossingRate(buffer);
      const energy = this.energy(freq);

      const s = Math.max(0, Math.min(100, ((snr + 10) / 40) * 100));
      const z = Math.max(0, (1 - Math.min(zcr / 0.3, 1)) * 100);
      const c = Math.max(0, Math.min(100, (centroid / 250) * 100));
      const e = Math.max(0, Math.min(100, energy));

      clarity = Math.round(s * 0.6 + z * 0.2 + c * 0.1 + e * 0.1); // ← Starts at 0
    }

    // Voice stability
    const pitchMean = this.pitchHistory.reduce((a, b) => a + b, 0) / this.pitchHistory.length || 0;
    const pitchVar = Math.sqrt(this.pitchHistory.reduce((a, b) => a + (b - pitchMean) ** 2, 0) / this.pitchHistory.length || 0);
    const voiceStability = Math.max(0, Math.min(100, 100 - pitchVar * 2));

    // Speech rate
    let zc = 0;
    for (let i = 1; i < buffer.length; i++) if (buffer[i - 1] * buffer[i] < 0) zc++;
    const speechRate = (zc / buffer.length) * (this.audioContext!.sampleRate / 2) / 10;

    // Emotion
    let audioEmotion = 'neutral', audioConfidence = 0.5;
    if (volume > 60 && pitch > 200) { audioEmotion = 'excited'; audioConfidence = 0.8; }
    else if (volume < 30 && pitch < 150) { audioEmotion = 'calm'; audioConfidence = 0.7; }
    else if (pitchVar > 30) { audioEmotion = 'nervous'; audioConfidence = 0.6; }
    else if (volume > 50 && pitchVar < 10) { audioEmotion = 'confident'; audioConfidence = 0.75; }

    return {
      pitch: Math.round(pitch),
      volume: Math.round(volume),
      speechRate: Math.round(speechRate),
      voiceStability: Math.round(voiceStability),
      audioEmotion,
      audioConfidence,
      clarity, // ← Real 0–100 clarity
    };
  }

  private spectralCentroid(freq: Uint8Array): number {
    let sum = 0, weighted = 0;
    for (let i = 0; i < freq.length; i++) {
      const v = freq[i] / 255;
      weighted += i * v;
      sum += v;
    }
    return sum > 0 ? weighted / sum : 0;
  }

  private zeroCrossingRate(buffer: Float32Array): number {
    let count = 0;
    for (let i = 1; i < buffer.length; i++) {
      if ((buffer[i - 1] >= 0) !== (buffer[i] >= 0)) count++;
    }
    return count / buffer.length;
  }

  private energy(freq: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < freq.length; i++) {
      const v = freq[i] / 255;
      sum += v * v;
    }
    return Math.sqrt(sum / freq.length) * 100;
  }

  private yinPitchDetection(buffer: Float32Array, sampleRate: number): number {
    const size = buffer.length / 2;
    const yin = new Float32Array(size);
    let running = 0;

    yin[0] = 1;
    for (let t = 1; t < size; t++) {
      let sum = 0;
      for (let i = 0; i < size; i++) {
        const d = buffer[i] - buffer[i + t];
        sum += d * d;
      }
      yin[t] = sum;
      running += sum;
      yin[t] *= t / (running || 1);
    }

    for (let t = 2; t < size; t++) {
      if (yin[t] < this.YIN_THRESHOLD) {
        const better = t + (yin[t - 1] - yin[t + 1]) / (2 * (yin[t - 1] + yin[t + 1] - 2 * yin[t]));
        return sampleRate / better;
      }
    }
    return 0;
  }

  private getDefaultMetrics(): BodyLanguageMetrics {
    return {
      face: { eyeContact: 0, emotion: 'neutral', emotionConfidence: 0, facialMovement: 0, gazeDirection: { x: 0, y: 0 }, landmarks: [] },
      posture: { postureScore: 0, shoulderAlignment: 0, headPosition: 0, stability: 0, landmarks: [] },
      gestures: { gestureCount: 0, gestureVariety: 0, handVisibility: 0, movementPatterns: [] },
      audio: { pitch: 0, volume: 0, speechRate: 0, voiceStability: 0, audioEmotion: 'silent', audioConfidence: 0, clarity: 0 },
      overallConfidence: 0,
      timestamp: Date.now(),
    };
  }

  cleanup() {
    this.faceLandmarker?.close();
    this.poseLandmarker?.close();
    this.gestureRecognizer?.close();
    this.audioContext?.close();
    this.isInitialized = false;
    this.previousFaceLandmarks = null;
    this.previousPoseLandmarks = null;
    this.gestureHistory = [];
    this.emotionHistory = [];
    this.pitchHistory = [];
  }
}
