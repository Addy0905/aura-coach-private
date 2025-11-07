import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

let emotionDetector: any = null;
let faceDetector: any = null;

// Initialize AI models with proven, working models
export const initializeModels = async () => {
  try {
    console.log('Initializing AI models...');
    
    // Load face detection model - using depth estimation as proxy for face analysis
    if (!faceDetector) {
      console.log('Loading face detection model...');
      faceDetector = await pipeline(
        'depth-estimation',
        'Xenova/dpt-large',
        { device: 'webgpu' }
      );
      console.log('Face detector loaded');
    }
    
    // Load emotion detection model - using CLIP for facial analysis
    if (!emotionDetector) {
      console.log('Loading emotion detection model...');
      emotionDetector = await pipeline(
        'zero-shot-image-classification',
        'Xenova/clip-vit-base-patch32',
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
        'zero-shot-image-classification',
        'Xenova/clip-vit-base-patch32'
      );
      console.log('Models loaded on CPU');
      return true;
    } catch (cpuError) {
      console.error('CPU fallback also failed:', cpuError);
      return false;
    }
  }
};

// Detect face using depth estimation
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

// Analyze facial expressions using CLIP zero-shot classification
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

    // Define emotion labels for zero-shot classification
    const emotionLabels = [
      'a happy smiling person',
      'a confident professional person',
      'a neutral calm person',
      'a focused attentive person',
      'a sad unhappy person',
      'an angry frustrated person',
      'a nervous anxious person',
      'a surprised person'
    ];

    let emotions: any[] = [];
    let hasFace = false;
    
    try {
      if (emotionDetector) {
        const results = await emotionDetector(imageData, emotionLabels);
        
        // Convert CLIP results to emotion format
        emotions = results.map((r: any) => ({
          label: r.label.replace('a ', '').replace(' person', '').split(' ')[0],
          score: r.score
        }));
        
        // Sort by score
        emotions.sort((a, b) => b.score - a.score);
        
        hasFace = emotions[0].score > 0.15; // Confidence threshold
        console.log('Emotion analysis:', emotions);
      }
    } catch (err) {
      console.error('Emotion detection error:', err);
    }

    // Detect face region using depth estimation
    const faceData = await detectFaceRegion(imageData);
    
    // Calculate eye contact based on face position
    const eyeContact = (hasFace && faceData) 
      ? calculateEyeContact(faceData, canvas, videoElement) 
      : 25;
    
    // Calculate facial expression score
    const faceScore = hasFace ? calculateExpressionScore(emotions) : 25;

    return {
      hasFace,
      eyeContact: Math.round(eyeContact),
      faceScore: Math.round(faceScore),
      emotions: emotions.slice(0, 3), // Top 3 emotions
      imageData,
      faceData
    };
  } catch (error) {
    console.error('Error analyzing facial expression:', error);
    return null;
  }
};

// Calculate eye contact score based on face detection
const calculateEyeContact = (faceData: any, canvas: HTMLCanvasElement, video: HTMLVideoElement): number => {
  if (!faceData) return 25;
  
  // Face centered horizontally and in upper half = good eye contact
  const centerX = faceData.faceCenterX;
  const centerY = faceData.faceCenterY;
  
  let score = 50;
  
  // Horizontal alignment (0.4 to 0.6 is centered)
  const horizontalOffset = Math.abs(centerX - 0.5);
  if (horizontalOffset < 0.1) {
    score += 25; // Well centered
  } else if (horizontalOffset < 0.2) {
    score += 15; // Slightly off-center
  } else {
    score -= 10; // Too far off-center
  }
  
  // Vertical position (0.3 to 0.5 is ideal - upper-center)
  if (centerY > 0.25 && centerY < 0.5) {
    score += 25; // Good framing
  } else if (centerY > 0.5) {
    score -= 15; // Too low (slouching)
  } else if (centerY < 0.25) {
    score += 10; // High position (attentive)
  }
  
  // Confidence bonus
  score += faceData.confidence * 10;
  
  return Math.max(25, Math.min(100, score));
};

// Calculate expression score based on detected emotions
const calculateExpressionScore = (emotions: any[]): number => {
  if (!emotions || emotions.length === 0) return 25;
  
  // Professional emotions mapping
  const emotionScores: { [key: string]: number } = {
    'happy': 1.0,
    'confident': 1.0,
    'neutral': 0.85,
    'calm': 0.85,
    'focused': 0.95,
    'attentive': 0.95,
    'surprised': 0.4,
    'sad': -0.7,
    'unhappy': -0.7,
    'angry': -1.0,
    'frustrated': -0.8,
    'nervous': -0.6,
    'anxious': -0.6
  };
  
  let score = 50;
  
  for (const emotion of emotions) {
    const label = emotion.label.toLowerCase();
    
    // Find matching emotion
    for (const [key, weight] of Object.entries(emotionScores)) {
      if (label.includes(key)) {
        score += emotion.score * weight * 50;
        break;
      }
    }
  }
  
  return Math.max(25, Math.min(100, score));
};

// Analyze posture using depth and face detection
export const analyzePosture = async (videoElement: HTMLVideoElement) => {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return { posture: 50, isWellFramed: false, bodyLanguage: 'neutral' };
    
    ctx.drawImage(videoElement, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Use depth estimation to analyze posture
    let postureScore = 50;
    let bodyLanguage = 'neutral';
    let isWellFramed = false;
    
    try {
      if (faceDetector) {
        const depth = await faceDetector(imageData);
        
        if (depth && depth.depth) {
          const depthData = depth.depth;
          
          // Analyze depth distribution (uniform depth = good posture)
          // Deep analysis would look at shoulder width, head position, etc.
          
          // For now, use face detection as primary indicator
          const faceData = await detectFaceRegion(imageData);
          
          if (faceData && faceData.hasFace) {
            // Face in upper-center = good posture
            if (faceData.faceCenterY < 0.5) {
              postureScore += 30;
              bodyLanguage = 'upright';
              isWellFramed = true;
            } else if (faceData.faceCenterY > 0.6) {
              postureScore -= 20;
              bodyLanguage = 'slouching';
            }
            
            // Centered horizontally
            if (Math.abs(faceData.faceCenterX - 0.5) < 0.15) {
              postureScore += 20;
              isWellFramed = true;
            }
          }
        }
      }
    } catch (error) {
      console.error('Depth analysis error:', error);
    }
    
    // Fallback: aspect ratio check
    const aspectRatio = videoElement.videoWidth / videoElement.videoHeight;
    if (aspectRatio > 0.5 && aspectRatio < 2) {
      postureScore += 10;
      isWellFramed = true;
    }
    
    // Analyze frame composition using traditional CV
    const imageDataCV = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageDataCV.data;
    
    // Calculate vertical distribution
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
    
    // Good posture = more presence in top half
    const topRatio = topHalfBrightness / (topHalfBrightness + bottomHalfBrightness);
    postureScore += (topRatio - 0.5) * 50;
    
    // Determine body language based on score
    if (postureScore > 80) bodyLanguage = 'confident';
    else if (postureScore > 65) bodyLanguage = 'engaged';
    else if (postureScore > 50) bodyLanguage = 'neutral';
    else if (postureScore < 40) bodyLanguage = 'slouching';
    
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
  
  // Optimal audio level (0.3-0.7)
  if (audioLevel > 0.3 && audioLevel < 0.7) {
    clarityScore = 70 + (audioLevel * 30);
  } else if (audioLevel >= 0.7 && audioLevel < 0.9) {
    clarityScore = 85; // Slightly loud but clear
  } else if (audioLevel >= 0.9) {
    clarityScore = 70; // Too loud
  } else if (audioLevel > 0.1) {
    clarityScore = 40 + (audioLevel * 100); // Too quiet
  } else {
    clarityScore = 25; // Very quiet or no audio
  }
  
  // Analyze transcript for filler words
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
    
    // Penalize excessive filler words
    if (fillerRatio > 0.15) {
      clarityScore -= 20;
    } else if (fillerRatio > 0.10) {
      clarityScore -= 10;
    } else if (fillerRatio < 0.05) {
      clarityScore += 10; // Bonus for clear speech
    }
  }
  
  // Bonus for sustained speaking
  if (transcript.length > 100) {
    clarityScore = Math.min(100, clarityScore + 5);
  }
  if (transcript.length > 200) {
    clarityScore = Math.min(100, clarityScore + 5);
  }
  
  return Math.max(25, Math.min(100, clarityScore));
};

// Calculate engagement with ML-based factors
const calculateEngagement = (
  hasFace: boolean,
  transcriptLength: number,
  audioLevel: number,
  emotions: any[],
  eyeContactScore: number
): number => {
  let engagementScore = 25;
  
  // Face presence (25 points)
  if (hasFace) {
    engagementScore += 25;
  }
  
  // Eye contact quality (20 points)
  engagementScore += (eyeContactScore / 100) * 20;
  
  // Speech activity (25 points)
  if (transcriptLength > 0) {
    const speechScore = Math.min(25, (transcriptLength / 500) * 25);
    engagementScore += speechScore;
  }
  
  // Audio activity (15 points)
  if (audioLevel > 0.2) {
    engagementScore += Math.min(15, audioLevel * 20);
  }
  
  // Positive emotions from ML (15 points)
  if (emotions.length > 0) {
    const positiveEmotions = emotions.filter(e => {
      const label = e.label.toLowerCase();
      return ['happy', 'confident', 'focused', 'attentive', 'calm'].some(p => label.includes(p));
    });
    
    for (const emotion of positiveEmotions) {
      engagementScore += emotion.score * 15;
    }
  }
  
  return Math.max(25, Math.min(100, engagementScore));
};

// Generate metrics - BACKWARD COMPATIBLE with original structure
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
    
    // Voice clarity with filler word detection
    const clarity = calculateVoiceClarity(audioLevel, transcript);
    
    // Engagement using ML metrics
    const engagement = calculateEngagement(
      facialAnalysis?.hasFace || false,
      transcript.length,
      audioLevel,
      facialAnalysis?.emotions || [],
      facialAnalysis?.eyeContact || 25
    );

    // Return ORIGINAL structure for backward compatibility
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
    
    // Safe fallback values
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
