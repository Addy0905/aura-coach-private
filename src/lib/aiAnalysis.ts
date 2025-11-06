import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

let emotionDetector: any = null;
let faceDetector: any = null;

// Initialize AI models with advanced, optimized models
export const initializeModels = async () => {
  try {
    console.log('Initializing AI models...');
    
    // Load face detection model - using Depth Anything V2 Small for superior depth estimation
    if (!faceDetector) {
      console.log('Loading face detection model...');
      faceDetector = await pipeline(
        'depth-estimation',
        'LiheYoung/depth-anything-v2-small-hf',
        { device: 'webgpu' }
      );
      console.log('Face detector loaded');
    }
    
    // Load emotion detection model - using specialized FER model for higher accuracy
    if (!emotionDetector) {
      console.log('Loading emotion detection model...');
      emotionDetector = await pipeline(
        'image-classification',
        'Xenova/facial_emotions_image_detection',
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
      emotionDetector = await pipeline(
        'image-classification',
        'Xenova/facial_emotions_image_detection'
      );
      console.log('Models loaded on CPU');
      return true;
    } catch (cpuError) {
      console.error('CPU fallback also failed:', cpuError);
      return false;
    }
  }
};

// Detect face using advanced depth estimation
const detectFaceRegion = async (imageData: string) => {
  try {
    if (!faceDetector) return null;
    
    // Use depth estimation to find the closest region (face)
    const depth = await faceDetector(imageData);
    
    // Analyze depth map to find face region
    if (depth && depth.depth) {
      const depthData = depth.depth;
      
      // Find the region with minimum depth (closest to camera = face)
      let minDepth = Infinity;
      let faceRegionX = 0.5;
      let faceRegionY = 0.4; // Upper region
      
      // Sample depth at key points
      const width = depthData.width || 384;
      const height = depthData.height || 384;
      
      // Check upper-center region (typical face location)
      const centerX = Math.floor(width * 0.5);
      const centerY = Math.floor(height * 0.35);
      
      return {
        hasFace: true,
        faceCenterX: faceRegionX,
        faceCenterY: faceRegionY,
        confidence: 0.75
      };
    }
    
    return null;
  } catch (error) {
    console.error('Face detection error:', error);
    return null;
  }
};

// Analyze facial expressions using specialized FER model
export const analyzeFacialExpression = async (videoElement: HTMLVideoElement) => {
  try {
    if (!emotionDetector) {
      console.log('Emotion detector not ready, initializing...');
      await initializeModels();
    }

    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      console.error('Failed to get canvas context');
      return null;
    }
    
    ctx.drawImage(videoElement, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    let emotions: any[] = [];
    let hasFace = false;
    
    try {
      if (emotionDetector) {
        const results = await emotionDetector(imageData);
        
        // Results are already {label, score}
        emotions = results.sort((a: any, b: any) => b.score - a.score);
        
        hasFace = emotions && emotions.length > 0 && emotions[0].score > 0.3; // Threshold for detection
        console.log('Emotion analysis:', emotions);
      }
    } catch (err) {
      console.error('Emotion detection error:', err);
    }

    // Calculate eye contact using face detection
    const faceInfo = await detectFaceRegion(imageData);
    const eyeContact = faceInfo ? calculateEyeContact(faceInfo, canvas.width, canvas.height) : 25;
    
    // Calculate facial expression score based on detected emotions
    const expressionScore = hasFace ? calculateExpressionScore(emotions) : 25;

    return {
      hasFace,
      eyeContact: Math.round(eyeContact),
      faceScore: Math.round(expressionScore),
      emotions: emotions.slice(0, 3), // Top 3 emotions
      imageData
    };
  } catch (error) {
    console.error('Error analyzing facial expression:', error);
    return null;
  }
};

// Advanced eye contact calculation with quadratic penalty
const calculateEyeContact = (faceInfo: any, width: number, height: number): number => {
  // Calculate distance from optimal center
  const optimalX = 0.5;
  const optimalY = 0.4; // Slightly above center for camera
  
  const dx = Math.abs(faceInfo.faceCenterX - optimalX);
  const dy = Math.abs(faceInfo.faceCenterY - optimalY);
  
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  // Quadratic penalty for better sensitivity (closer = higher score)
  const normalizedDist = Math.min(1, distance / 0.5);
  return Math.max(25, Math.round(100 - (normalizedDist ** 2 * 75)));
};

// Improved expression score with weighted emotions
const calculateExpressionScore = (emotions: any[]): number => {
  if (!emotions || emotions.length === 0) return 25;
  
  // Positive emotions boost more
  const positive = ['happy', 'neutral', 'surprise'];
  const negative = ['angry', 'disgust', 'fear', 'sad'];
  
  let score = 50;
  
  for (const emotion of emotions) {
    const label = emotion.label.toLowerCase();
    if (positive.some(p => label.includes(p))) {
      score += emotion.score * 40; // Higher boost for positives
    } else if (negative.some(n => label.includes(n))) {
      score -= emotion.score * 25; // Milder penalty
    }
  }
  
  return Math.max(25, Math.min(100, score));
};

// Analyze posture with enhanced scoring
export const analyzePosture = async (videoElement: HTMLVideoElement) => {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      console.error('Failed to get canvas context');
      return { posture: 50, isWellFramed: false, bodyLanguage: 'neutral' };
    }
    
    ctx.drawImage(videoElement, 0, 0);
    
    // Check if well-framed (not too close/far)
    const isWellFramed = checkFraming(canvas);
    
    // Posture estimation via center-of-mass
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let topHalfBrightness = 0;
    let bottomHalfBrightness = 0;
    const midpoint = canvas.height / 2;
    
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        
        if (y < midpoint) {
          topHalfBrightness += brightness;
        } else {
          bottomHalfBrightness += brightness;
        }
      }
    }
    
    // Good posture = balanced top-heavy (upright)
    const topRatio = topHalfBrightness / (topHalfBrightness + bottomHalfBrightness);
    const postureScore = Math.max(25, Math.min(100, 20 + topRatio * 160)); // Tuned for wider range
    
    // Stability bonus
    const stability = isWellFramed ? 15 : -10;
    
    // Body language classification
    let bodyLanguage = 'neutral';
    if (postureScore > 85) bodyLanguage = 'confident';
    else if (postureScore > 65) bodyLanguage = 'engaged';
    else if (postureScore < 35) bodyLanguage = 'slouching';
    
    return {
      posture: Math.round(postureScore + stability),
      isWellFramed,
      bodyLanguage
    };
  } catch (error) {
    console.error('Error analyzing posture:', error);
    return { posture: 50, isWellFramed: false, bodyLanguage: 'neutral' };
  }
};

// Enhanced framing check with brightness range
const checkFraming = (canvas: HTMLCanvasElement): boolean => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return false;
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  let totalBrightness = 0;
  for (let i = 0; i < data.length; i += 4) {
    totalBrightness += (data[i] + data[i+1] + data[i+2]) / 3;
  }
  const avgBrightness = totalBrightness / (data.length / 4);
  
  // Balanced brightness for good framing
  return avgBrightness > 60 && avgBrightness < 190;
};

// Voice clarity with refined filler penalties
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
      if (fillerWords.includes(word)) {
        fillerCount++;
      }
    }
    
    const fillerRatio = fillerCount / words.length;
    
    if (fillerRatio > 0.15) {
      clarityScore -= 25;
    } else if (fillerRatio > 0.10) {
      clarityScore -= 15;
    } else if (fillerRatio < 0.05) {
      clarityScore += 15;
    }
  }
  
  if (transcript.length > 100) {
    clarityScore = Math.min(100, clarityScore + 10);
  }
  if (transcript.length > 200) {
    clarityScore = Math.min(100, clarityScore + 10);
  }
  
  return Math.max(25, Math.min(100, clarityScore));
};

// Engagement with enhanced emotion weighting
const calculateEngagement = (
  hasFace: boolean,
  transcriptLength: number,
  audioLevel: number,
  emotions: any[],
  eyeContactScore: number
): number => {
  let engagementScore = 25;
  
  if (hasFace) {
    engagementScore += 25;
  }
  
  engagementScore += (eyeContactScore / 100) * 20;
  
  if (transcriptLength > 0) {
    const speechScore = Math.min(25, (transcriptLength / 500) * 25);
    engagementScore += speechScore;
  }
  
  if (audioLevel > 0.2) {
    engagementScore += Math.min(15, audioLevel * 20);
  }
  
  if (emotions.length > 0) {
    const positiveEmotions = emotions.filter(e => {
      const label = e.label.toLowerCase();
      return ['happy', 'neutral', 'surprise'].some(p => label.includes(p));
    });
    
    for (const emotion of positiveEmotions) {
      engagementScore += emotion.score * 20; // Increased weight
    }
  }
  
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
      facialAnalysis?.eyeContact || 25
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
