import { pipeline, env } from '@huggingface/transformers';
import { FaceLandmarker, PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

let emotionDetector: any = null;
let faceLandmarker: any = null;
let poseLandmarker: any = null;

// Initialize AI models with advanced, accurate models
export const initializeModels = async () => {
  try {
    console.log('Initializing AI models...');

    // Initialize MediaPipe FilesetResolver
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    // Load advanced Face Landmarker from MediaPipe for precise face mesh and blendshapes
    if (!faceLandmarker) {
      console.log('Loading Face Landmarker...');
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU"  // Use GPU if available
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: 0.6,  // Raised for higher accuracy
        minFacePresenceConfidence: 0.6,
        minTrackingConfidence: 0.6
      });
      console.log('Face Landmarker loaded');
    }

    // Load Pose Landmarker from MediaPipe for accurate posture analysis
    if (!poseLandmarker) {
      console.log('Loading Pose Landmarker...');
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.6,
        minPosePresenceConfidence: 0.6,
        minTrackingConfidence: 0.6
      });
      console.log('Pose Landmarker loaded');
    }

    // Load advanced emotion detection model - using high-accuracy ViT model (~84% on evaluation set)
    if (!emotionDetector) {
      console.log('Loading emotion detection model...');
      emotionDetector = await pipeline(
        'image-classification',
        'mo-thecreator/vit-Facial-Expression-Recognition',  // High-accuracy model for emotions
        { device: 'webgpu' }
      );
      console.log('Emotion detector loaded');
    }

    console.log('AI models initialized successfully');
    return true;
  } catch (error) {
    console.error('Error initializing AI models:', error);
    // Fallback to CPU if WebGPU fails
    try {
      console.log('Retrying with CPU...');
      emotionDetector = await pipeline('image-classification', 'mo-thecreator/vit-Facial-Expression-Recognition');
      // MediaPipe already has delegate fallback
      console.log('Models loaded on CPU');
      return true;
    } catch (cpuError) {
      console.error('CPU fallback also failed:', cpuError);
      return false;
    }
  }
};

// Helper to create canvas from video
const createCanvasFromVideo = (videoElement: HTMLVideoElement) => {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }
  ctx.drawImage(videoElement, 0, 0);
  return { canvas, ctx };
};

// Helper to crop image to face bounding box using canvas
const cropToFace = (canvas: HTMLCanvasElement, landmarks: any[]) => {
  if (!landmarks || landmarks.length === 0) return canvas.toDataURL('image/jpeg', 0.8);

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
  if (!cropCtx) return canvas.toDataURL('image/jpeg', 0.8);

  cropCtx.drawImage(
    canvas,
    pxMinX, pxMinY, pxWidth, pxHeight,  // Source rect
    0, 0, pxWidth, pxHeight              // Dest rect
  );

  return cropCanvas.toDataURL('image/jpeg', 0.8);
};

// Analyze facial expressions using MediaPipe for landmarks and HF for emotions
export const analyzeFacialExpression = async (videoElement: HTMLVideoElement) => {
  try {
    if (!faceLandmarker || !emotionDetector) {
      console.log('Models not ready, initializing...');
      await initializeModels();
    }

    const { canvas } = createCanvasFromVideo(videoElement);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const timestamp = performance.now();  // For video mode

    // Detect face with MediaPipe
    const faceResults = await faceLandmarker.detectForVideo(videoElement, timestamp);

    let emotions: any[] = [];
    let hasFace = false;
    let faceData: any = null;
    let blendshapes: any[] = [];

    if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
      hasFace = true;
      faceData = {
        landmarks: faceResults.faceLandmarks[0],
        transformationMatrix: faceResults.facialTransformationMatrixes?.[0],
        centerX: faceResults.faceLandmarks[0].reduce((sum, lm) => sum + lm.x, 0) / faceResults.faceLandmarks[0].length,
        centerY: faceResults.faceLandmarks[0].reduce((sum, lm) => sum + lm.y, 0) / faceResults.faceLandmarks[0].length,
        confidence: 1.0  // MediaPipe confidence
      };
      blendshapes = faceResults.faceBlendshapes?.[0]?.categories || [];

      // Crop to face for accurate emotion detection
      const croppedImage = cropToFace(canvas, faceResults.faceLandmarks[0]);

      // Run emotion detection on cropped face
      const results = await emotionDetector(croppedImage);
      emotions = results.map((r: any) => ({ label: r.label, score: r.score }));
      emotions = emotions.filter(e => e.score > 0.5);  // Filter low-confidence
      emotions.sort((a, b) => b.score - a.score);
    }

    // Calculate eye contact using head pose from transformation matrix or landmarks
    const eyeContact = hasFace ? calculateEyeContact(faceData, blendshapes) : 25;

    // Calculate facial expression score using emotions and blendshapes
    const faceScore = hasFace ? calculateExpressionScore(emotions, blendshapes) : 25;

    return {
      hasFace,
      eyeContact: Math.round(eyeContact),
      faceScore: Math.round(faceScore),
      emotions: emotions.slice(0, 3),  // Top 3 emotions
      imageData,
      faceData
    };
  } catch (error) {
    console.error('Error analyzing facial expression:', error);
    return null;
  }
};

// Updated eye contact calculation using MediaPipe landmarks/blendshapes
const calculateEyeContact = (faceData: any, blendshapes: any[]): number => {
  if (!faceData) return 25;

  let score = 50;

  // Use center for basic positioning
  const horizontalOffset = Math.abs(faceData.centerX - 0.5);
  if (horizontalOffset < 0.05) score += 25;
  else if (horizontalOffset < 0.15) score += 15;
  else score -= 15;

  if (faceData.centerY > 0.3 && faceData.centerY < 0.45) score += 25;
  else if (faceData.centerY > 0.5) score -= 20;
  else if (faceData.centerY < 0.3) score += 10;

  // Advanced: Use blendshapes for eye look (e.g., eyesLookLeft/Right)
  const eyeLookLeft = blendshapes.find(b => b.categoryName === 'eyesLookLeft')?.score || 0;
  const eyeLookRight = blendshapes.find(b => b.categoryName === 'eyesLookRight')?.score || 0;
  const gazeOffset = Math.abs(eyeLookLeft - eyeLookRight);
  if (gazeOffset < 0.2) score += 20;  // Centered gaze
  else score -= 10;

  return Math.max(25, Math.min(100, score));
};

// Updated expression score with blendshapes for accuracy
const calculateExpressionScore = (emotions: any[], blendshapes: any[]): number => {
  if (!emotions || emotions.length === 0) return 25;

  const emotionScores: { [key: string]: number } = {
    'happy': 1.0,
    'neutral': 0.85,
    'surprise': 0.3,
    'sad': -0.8,
    'angry': -1.0,
    'disgust': -0.9,
    'fear': -0.7
  };

  let score = 50;
  for (const emotion of emotions) {
    const label = emotion.label.toLowerCase();
    const weight = emotionScores[label] || 0;
    score += emotion.score * weight * 60;
  }

  // Enhance with blendshapes (e.g., smile, frown)
  const smileScore = (blendshapes.find(b => b.categoryName === 'mouthSmileLeft')?.score || 0) + (blendshapes.find(b => b.categoryName === 'mouthSmileRight')?.score || 0) / 2;
  const frownScore = (blendshapes.find(b => b.categoryName === 'mouthFrownLeft')?.score || 0) + (blendshapes.find(b => b.categoryName === 'mouthFrownRight')?.score || 0) / 2;
  score += smileScore * 20 - frownScore * 20;

  return Math.max(25, Math.min(100, score));
};

// Analyze posture using MediaPipe Pose Landmarker
export const analyzePosture = async (videoElement: HTMLVideoElement) => {
  try {
    if (!poseLandmarker) {
      await initializeModels();
    }

    const timestamp = performance.now();
    const poseResults = await poseLandmarker.detectForVideo(videoElement, timestamp);

    let postureScore = 50;
    let bodyLanguage = 'neutral';
    let isWellFramed = false;

    if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
      const landmarks = poseResults.landmarks[0];
      const worldLandmarks = poseResults.worldLandmarks?.[0] || [];

      // Calculate spine alignment (shoulders to hips)
      const leftShoulder = landmarks[11];
      const rightShoulder = landmarks[12];
      const leftHip = landmarks[23];
      const rightHip = landmarks[24];

      if (leftShoulder && rightShoulder && leftHip && rightHip) {
        const shoulderMid = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
        const hipMid = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2 };
        const verticalDiff = Math.abs(shoulderMid.y - hipMid.y);
        const horizontalOffset = Math.abs(shoulderMid.x - hipMid.x);

        if (verticalDiff > 0.3 && horizontalOffset < 0.1) {  // Upright
          postureScore += 30;
          bodyLanguage = 'upright';
          isWellFramed = true;
        } else if (horizontalOffset > 0.15) {
          postureScore -= 25;
          bodyLanguage = 'slouching';
        }

        // Centering
        if (Math.abs(shoulderMid.x - 0.5) < 0.12) postureScore += 25;

        // Use 3D for depth balance
        if (worldLandmarks.length > 0) {
          const leftShoulderZ = worldLandmarks[11].z;
          const rightShoulderZ = worldLandmarks[12].z;
          const balance = Math.abs(leftShoulderZ - rightShoulderZ);
          if (balance < 0.05) postureScore += 20;
          else postureScore -= 10;
        }
      }
    } else {
      // Fallback to basic analysis if no pose detected
      const { canvas, ctx } = createCanvasFromVideo(videoElement);
      const imageDataCV = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageDataCV.data;
      let topHalfBrightness = 0;
      let bottomHalfBrightness = 0;
      const midpoint = canvas.height / 2;
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const idx = (y * canvas.width + x) * 4;
          const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          if (y < midpoint) topHalfBrightness += brightness;
          else bottomHalfBrightness += brightness;
        }
      }
      const topRatio = topHalfBrightness / (topHalfBrightness + bottomHalfBrightness + 0.001);
      postureScore += (topRatio - 0.5) * 60;
    }

    // Determine body language
    if (postureScore > 85) bodyLanguage = 'confident';
    else if (postureScore > 70) bodyLanguage = 'engaged';
    else if (postureScore > 50) bodyLanguage = 'neutral';
    else bodyLanguage = 'slouching';

    return {
      posture: Math.round(Math.max(25, Math.min(100, postureScore))),
      isWellFramed,
      bodyLanguage
    };
  } catch (error) {
    console.error('Error analyzing posture:', error);
    return { posture: 50, isWellFramed: false, bodyLanguage: 'neutral' };
  }
};

// Calculate voice clarity with filler word detection
const calculateVoiceClarity = (audioLevel: number, transcript: string): number => {
  let clarityScore = 50;
  if (audioLevel > 0.3 && audioLevel < 0.7) {
    clarityScore = 70 + (audioLevel * 30);
  } else if (audioLevel >= 0.7 && audioLevel < 0.9) {
    clarityScore = 85;
  } else if (audioLevel >= 0.9) {
    clarityScore = 70;
  } else if (audioLevel > 0.1) {
    clarityScore = 40 + (audioLevel * 100);
  } else {
    clarityScore = 25;
  }
  if (transcript.length > 50) {
    const words = transcript.toLowerCase().split(/\s+/);
    const fillerWords = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally'];
    let fillerCount = 0;
    for (const word of words) {
      if (fillerWords.includes(word)) fillerCount++;
    }
    const fillerRatio = fillerCount / words.length;
    if (fillerRatio > 0.15) clarityScore -= 25;
    else if (fillerRatio > 0.10) clarityScore -= 15;
    else if (fillerRatio < 0.05) clarityScore += 15;
  }
  if (transcript.length > 100) clarityScore = Math.min(100, clarityScore + 10);
  if (transcript.length > 200) clarityScore = Math.min(100, clarityScore + 10);
  return Math.max(25, Math.min(100, clarityScore));
};

// Calculate engagement with ML-based factors
const calculateEngagement = (
  hasFace: boolean,
  transcriptLength: number,
  audioLevel: number,
  emotions: any[],
  eyeContactScore: number,
  blendshapes: any[] = []
): number => {
  let engagementScore = 25;
  if (hasFace) engagementScore += 25;
  engagementScore += (eyeContactScore / 100) * 20;
  if (transcriptLength > 0) {
    const speechScore = Math.min(25, (transcriptLength / 500) * 25);
    engagementScore += speechScore;
  }
  if (audioLevel > 0.2) engagementScore += Math.min(15, audioLevel * 20);
  if (emotions.length > 0) {
    const positiveEmotions = emotions.filter(e => {
      const label = e.label.toLowerCase();
      return ['happy', 'neutral', 'surprise'].some(p => label.includes(p));
    });
    for (const emotion of positiveEmotions) {
      engagementScore += emotion.score * 20;
    }
  }
  // Add blendshape-based engagement (e.g., smile for positive)
  const smileScore = (blendshapes.find(b => b.categoryName === 'mouthSmileLeft')?.score || 0) + (blendshapes.find(b => b.categoryName === 'mouthSmileRight')?.score || 0) / 2;
  engagementScore += smileScore * 10;

  return Math.max(25, Math.min(100, engagementScore));
};

// Generate metrics - backward compatible
export const generateMetrics = async (
  videoElement: HTMLVideoElement,
  audioLevel: number,
  transcript: string
) => {
  console.log('Generating ML-based metrics with audio level:', audioLevel, 'transcript length:', transcript.length);
  try {
    const facialAnalysis = await analyzeFacialExpression(videoElement);
    const postureAnalysis = await analyzePosture(videoElement);
    console.log('Facial analysis:', facialAnalysis);
    console.log('Posture analysis:', postureAnalysis);

    const clarity = calculateVoiceClarity(audioLevel, transcript);
    const engagement = calculateEngagement(
      facialAnalysis?.hasFace || false,
      transcript.length,
      audioLevel,
      facialAnalysis?.emotions || [],
      facialAnalysis?.eyeContact || 25,
      facialAnalysis?.faceData?.blendshapes || []  // Pass blendshapes if available
    );

    return {
      eyeContact: facialAnalysis?.eyeContact || 25,
      posture: postureAnalysis.posture,
      clarity: Math.round(clarity),
      engagement: Math.round(engagement),
      bodyLanguage: postureAnalysis.bodyLanguage,
      emotions: facialAnalysis?.emotions || [],
      imageData: facialAnalysis?.imageData
    };
  } catch (error) {
    console.error('Error generating metrics:', error);
    return {
      eyeContact: 25,
      posture: 50,
      clarity: 50,
      engagement: 25,
      bodyLanguage: 'neutral',
      emotions: [],
      imageData: null
    };
  }
};
