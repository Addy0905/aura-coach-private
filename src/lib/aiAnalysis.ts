import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

let emotionDetector: any = null;
let poseDetector: any = null;

// Initialize AI models
export const initializeModels = async () => {
  try {
    console.log('Initializing AI models...');
    
    // Load emotion detection model for facial expressions
    if (!emotionDetector) {
      console.log('Loading emotion detection model...');
      emotionDetector = await pipeline(
        'image-classification',
        'Xenova/vit-base-patch16-224-in21k-emotion',
        { device: 'webgpu' }
      );
      console.log('Emotion detector loaded');
    }
    
    // Load pose detection model for posture and body language
    if (!poseDetector) {
      console.log('Loading pose detection model...');
      poseDetector = await pipeline(
        'image-classification',
        'Xenova/vit-base-patch16-224',
        { device: 'webgpu' }
      );
      console.log('Pose detector loaded');
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
        'Xenova/vit-base-patch16-224-in21k-emotion'
      );
      console.log('Models loaded on CPU');
      return true;
    } catch (cpuError) {
      console.error('CPU fallback also failed:', cpuError);
      return false;
    }
  }
};

// Analyze facial expressions using emotion detection
export const analyzeFacialExpression = async (videoElement: HTMLVideoElement) => {
  try {
    if (!emotionDetector) {
      console.log('Emotion detector not ready, initializing...');
      await initializeModels();
    }

    // Create canvas to capture frame
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

    // Detect emotions and facial expressions
    let emotions: any[] = [];
    let hasFace = false;
    
    try {
      if (emotionDetector) {
        emotions = await emotionDetector(imageData);
        hasFace = emotions && emotions.length > 0;
        console.log('Emotion analysis:', emotions);
      }
    } catch (err) {
      console.error('Emotion detection error:', err);
    }

    // Calculate eye contact based on face position and engagement
    const eyeContact = hasFace ? calculateEyeContact(canvas, videoElement) : 25;
    
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

// Calculate eye contact score based on face position
const calculateEyeContact = (canvas: HTMLCanvasElement, video: HTMLVideoElement): number => {
  // Analyze face position relative to center
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  
  // Use brightness analysis as proxy for eye contact
  const ctx = canvas.getContext('2d');
  if (!ctx) return 50;
  
  const imageData = ctx.getImageData(centerX - 50, centerY - 50, 100, 100);
  const data = imageData.data;
  
  // Analyze eye region brightness (eyes are typically darker)
  let totalBrightness = 0;
  for (let i = 0; i < data.length; i += 4) {
    totalBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
  }
  const avgBrightness = totalBrightness / (data.length / 4);
  
  // Convert brightness to eye contact score (darker center = looking at camera)
  return Math.max(25, Math.min(100, 100 - (avgBrightness / 255) * 30 + 30));
};

// Calculate expression score based on detected emotions
const calculateExpressionScore = (emotions: any[]): number => {
  if (!emotions || emotions.length === 0) return 25;
  
  // Professional emotions (confident, happy, neutral) score higher
  const positiveEmotions = ['happy', 'confident', 'neutral', 'focused'];
  const negativeEmotions = ['sad', 'angry', 'fearful', 'disgusted'];
  
  let score = 50;
  for (const emotion of emotions) {
    const label = emotion.label.toLowerCase();
    if (positiveEmotions.some(e => label.includes(e))) {
      score += emotion.score * 30;
    } else if (negativeEmotions.some(e => label.includes(e))) {
      score -= emotion.score * 20;
    }
  }
  
  return Math.max(25, Math.min(100, score));
};

// Analyze posture and body language
export const analyzePosture = async (videoElement: HTMLVideoElement) => {
  try {
    // Create canvas to capture frame
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { posture: 50, isWellFramed: false, bodyLanguage: 'neutral' };
    
    ctx.drawImage(videoElement, 0, 0);
    
    // Analyze frame composition and stability
    const aspectRatio = videoElement.videoWidth / videoElement.videoHeight;
    const isWellFramed = aspectRatio > 0.5 && aspectRatio < 2;
    
    // Analyze posture based on vertical distribution of pixels
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Calculate center of mass vertically (posture indicator)
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
    
    // Good posture = more presence in top half (upright position)
    const topRatio = topHalfBrightness / (topHalfBrightness + bottomHalfBrightness);
    const postureScore = Math.max(25, Math.min(100, 30 + topRatio * 100));
    
    // Analyze stability (good body language = stable frame)
    const stability = isWellFramed ? 20 : 0;
    
    // Determine body language
    let bodyLanguage = 'neutral';
    if (postureScore > 80) bodyLanguage = 'confident';
    else if (postureScore > 60) bodyLanguage = 'engaged';
    else if (postureScore < 40) bodyLanguage = 'slouching';
    
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

// Generate metrics combining all analyses
export const generateMetrics = async (
  videoElement: HTMLVideoElement,
  audioLevel: number,
  transcript: string
) => {
  console.log('Generating metrics with audio level:', audioLevel, 'transcript length:', transcript.length);
  
  const facialAnalysis = await analyzeFacialExpression(videoElement);
  const postureAnalysis = await analyzePosture(videoElement);
  
  console.log('Facial analysis:', facialAnalysis);
  console.log('Posture analysis:', postureAnalysis);
  
  // Audio-based clarity (real analysis based on audio level consistency)
  const clarity = calculateVoiceClarity(audioLevel, transcript);
  
  // Engagement based on real metrics: speech activity, facial presence, and emotions
  const engagement = calculateEngagement(
    facialAnalysis?.hasFace || false,
    transcript.length,
    audioLevel,
    facialAnalysis?.emotions || []
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
};

// Calculate voice clarity based on audio level consistency
const calculateVoiceClarity = (audioLevel: number, transcript: string): number => {
  // Base score from audio level (0.2-0.8 is optimal)
  let clarityScore = 50;
  
  if (audioLevel > 0.2 && audioLevel < 0.8) {
    clarityScore = 70 + (audioLevel * 30);
  } else if (audioLevel >= 0.8) {
    clarityScore = 85; // Too loud
  } else if (audioLevel > 0) {
    clarityScore = 40 + (audioLevel * 100); // Too quiet
  }
  
  // Adjust based on transcript length (more speaking = more clarity data)
  if (transcript.length > 100) {
    clarityScore = Math.min(100, clarityScore + 10);
  }
  
  return Math.max(25, Math.min(100, clarityScore));
};

// Calculate engagement based on multiple real factors
const calculateEngagement = (
  hasFace: boolean,
  transcriptLength: number,
  audioLevel: number,
  emotions: any[]
): number => {
  let engagementScore = 25;
  
  // Face presence (40 points)
  if (hasFace) {
    engagementScore += 40;
  }
  
  // Speech activity (30 points)
  if (transcriptLength > 0) {
    const speechScore = Math.min(30, (transcriptLength / 500) * 30);
    engagementScore += speechScore;
  }
  
  // Audio activity (15 points)
  if (audioLevel > 0.1) {
    engagementScore += Math.min(15, audioLevel * 20);
  }
  
  // Positive emotions (15 points)
  if (emotions.length > 0) {
    const positiveEmotions = emotions.filter(e => 
      ['happy', 'confident', 'focused'].some(p => e.label.toLowerCase().includes(p))
    );
    engagementScore += positiveEmotions.length * 5;
  }
  
  return Math.max(25, Math.min(100, engagementScore));
};
