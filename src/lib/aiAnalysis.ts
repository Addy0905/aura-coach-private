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
   
    // Load emotion detection model - using dedicated facial emotion classifier
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
// Detect face using depth estimation
const detectFaceRegion = async (imageData: string) => {
  try {
    if (!faceDetector) return null;
   
    // Use depth estimation to find the closest region (face)
    const depth = await faceDetector(imageData);
   
    // Analyze depth map to find face region
    if (depth && depth.predicted_depth) {
      const predictedDepth = depth.predicted_depth;
      const height = predictedDepth.dims[0];
      const width = predictedDepth.dims[1];
      const data = predictedDepth.data;
     
      // Find min and max depth (assuming lower depth value = closer)
      let minDepth = Infinity;
      let maxDepth = -Infinity;
      for (let i = 0; i < data.length; i++) {
        const val = data[i];
        if (val < minDepth) minDepth = val;
        if (val > maxDepth) maxDepth = val;
      }
     
      // Threshold for closest regions - tighter for face (0.3 -> 0.2 for better precision)
      const threshold = minDepth + (maxDepth - minDepth) * 0.2;
     
      // Compute centroid of closest pixels
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          if (data[idx] < threshold) {
            sumX += x;
            sumY += y;
            count++;
          }
        }
      }
     
      if (count === 0) return null;
     
      const faceCenterX = (sumX / count) / width;
      const faceCenterY = (sumY / count) / height;
     
      // Confidence based on size of closest region (face ~5-15% of frame for interviews)
      const areaFraction = count / (width * height);
      const confidence = Math.min(1, Math.max(0, (areaFraction - 0.05) / 0.1)); // Normalize between 0.05 to 0.15
      
      // Additional check: face should be in upper half
      const isValidFace = faceCenterY < 0.6 && areaFraction > 0.03 && areaFraction < 0.25;
     
      return {
        hasFace: isValidFace,
        faceCenterX,
        faceCenterY,
        confidence
      };
    }
   
    return null;
  } catch (error) {
    console.error('Face detection error:', error);
    return null;
  }
};
// Analyze facial expressions using dedicated emotion classifier
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
       
        // Convert results to emotion format
        emotions = results.map((r: any) => ({
          label: r.label,
          score: r.score
        }));
       
        // Sort by score descending
        emotions.sort((a, b) => b.score - a.score);
       
        hasFace = emotions[0].score > 0.25; // Higher confidence threshold for dedicated model
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
      : 0; // Change min to 0 for accuracy
   
    // Calculate facial expression score
    const faceScore = hasFace ? calculateExpressionScore(emotions) : 0;
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
  if (!faceData) return 0;
 
  // Face centered horizontally and in upper half = good eye contact
  const centerX = faceData.faceCenterX;
  const centerY = faceData.faceCenterY;
 
  let score = 50;
 
  // Horizontal alignment (0.45 to 0.55 is centered - stricter for accuracy)
  const horizontalOffset = Math.abs(centerX - 0.5);
  if (horizontalOffset < 0.05) {
    score += 25; // Well centered
  } else if (horizontalOffset < 0.15) {
    score += 15; // Slightly off-center
  } else {
    score -= 15; // Too far off-center - stronger penalty
  }
 
  // Vertical position (0.3 to 0.45 is ideal - upper-center)
  if (centerY > 0.3 && centerY < 0.45) {
    score += 25; // Good framing
  } else if (centerY > 0.5) {
    score -= 20; // Too low (slouching) - stronger penalty
  } else if (centerY < 0.3) {
    score += 10; // Slightly high (attentive)
  }
 
  // Confidence bonus
  score += faceData.confidence * 15; // Increased weight
 
  return Math.max(0, Math.min(100, score));
};
// Calculate expression score based on detected emotions
const calculateExpressionScore = (emotions: any[]): number => {
  if (!emotions || emotions.length === 0) return 0;
 
  // Professional emotions mapping - updated for new model labels
  const emotionScores: { [key: string]: number } = {
    'happy': 1.0,
    'confident': 1.0, // If model has, else map from happy/neutral
    'neutral': 0.85,
    'surprise': 0.3, // Lowered as surprise may indicate uncertainty
    'sad': -0.8, // Stronger penalty
    'angry': -1.0,
    'disgust': -0.9,
    'fear': -0.7
  };
 
  let score = 50;
 
  for (const emotion of emotions) {
    const label = emotion.label.toLowerCase();
   
    // Find matching emotion
    for (const [key, weight] of Object.entries(emotionScores)) {
      if (label.includes(key)) {
        score += emotion.score * weight * 60; // Increased multiplier for stronger differentiation
        break;
      }
    }
  }
 
  return Math.max(0, Math.min(100, score));
};
// Analyze posture using depth and face detection
export const analyzePosture = async (videoElement: HTMLVideoElement) => {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
   
    if (!ctx) return { posture: 0, isWellFramed: false, bodyLanguage: 'neutral' };
   
    ctx.drawImage(videoElement, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
   
    // Use depth estimation to analyze posture
    let postureScore = 50;
    let bodyLanguage = 'neutral';
    let isWellFramed = false;
   
    try {
      if (faceDetector) {
        const depth = await faceDetector(imageData);
       
        if (depth && depth.predicted_depth) {
          const predictedDepth = depth.predicted_depth;
          const height = predictedDepth.dims[0];
          const width = predictedDepth.dims[1];
          const data = predictedDepth.data;
         
          // Find min and max depth
          let minDepth = Infinity;
          let maxDepth = -Infinity;
          for (let i = 0; i < data.length; i++) {
            const val = data[i];
            if (val < minDepth) minDepth = val;
            if (val > maxDepth) maxDepth = val;
          }
         
          // Threshold for body region (broader than face)
          const bodyThreshold = minDepth + (maxDepth - minDepth) * 0.4;
         
          // Compute centroid and bounding box for body
          let sumX = 0;
          let sumY = 0;
          let count = 0;
          let minX = width, maxX = 0;
          let minY = height, maxY = 0;
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const idx = y * width + x;
              if (data[idx] < bodyThreshold) {
                sumX += x;
                sumY += y;
                count++;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
              }
            }
          }
         
          if (count > 0) {
            const bodyCenterX = (sumX / count) / width;
            const bodyCenterY = (sumY / count) / height;
            const bodyWidth = maxX - minX;
            const bodyHeight = maxY - minY;
            const aspectRatio = bodyHeight / (bodyWidth + 0.001); // Avoid division by zero
           
            // Posture based on aspect ratio (tall and narrow = upright)
            if (aspectRatio > 1.5) {
              postureScore += 30;
              bodyLanguage = 'upright';
              isWellFramed = true;
            } else if (aspectRatio < 1.0) {
              postureScore -= 25;
              bodyLanguage = 'slouching';
            }
           
            // Horizontal centering
            if (Math.abs(bodyCenterX - 0.5) < 0.12) {
              postureScore += 25;
              isWellFramed = true;
            } else if (Math.abs(bodyCenterX - 0.5) < 0.25) {
              postureScore += 10;
            } else {
              postureScore -= 15;
            }
           
            // Vertical position (body in upper 2/3)
            if (bodyCenterY < 0.6) {
              postureScore += 15;
            } else {
              postureScore -= 20;
            }
           
            // Symmetry check
            let leftCount = 0;
            let rightCount = 0;
            for (let y = minY; y <= maxY; y++) {
              for (let x = minX; x <= maxX; x++) {
                const idx = y * width + x;
                if (data[idx] < bodyThreshold) {
                  if (x < (minX + maxX) / 2) leftCount++;
                  else rightCount++;
                }
              }
            }
            const balance = Math.min(leftCount, rightCount) / (Math.max(leftCount, rightCount) + 0.001);
            if (balance > 0.85) {
              postureScore += 20;
            } else if (balance > 0.7) {
              postureScore += 10;
            } else {
              postureScore -= 10;
            }
          }
        }
      }
    } catch (error) {
      console.error('Depth analysis error:', error);
    }
   
    // Fallback: aspect ratio check
    const frameAspectRatio = videoElement.videoWidth / videoElement.videoHeight;
    if (frameAspectRatio > 0.5 && frameAspectRatio < 2) {
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
    const topRatio = topHalfBrightness / (topHalfBrightness + bottomHalfBrightness + 0.001);
    postureScore += (topRatio - 0.5) * 60; // Increased weight for better differentiation
   
    // Determine body language based on score
    if (postureScore > 85) bodyLanguage = 'confident';
    else if (postureScore > 70) bodyLanguage = 'engaged';
    else if (postureScore > 50) bodyLanguage = 'neutral';
    else bodyLanguage = 'slouching';
   
    return {
      posture: Math.round(Math.max(0, Math.min(100, postureScore))),
      isWellFramed,
      bodyLanguage
    };
  } catch (error) {
    console.error('Error analyzing posture:', error);
    return { posture: 0, isWellFramed: false, bodyLanguage: 'neutral' };
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
    clarityScore = 0; // Very quiet or no audio - set to 0
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
   
    // Penalize excessive filler words - stronger penalties
    if (fillerRatio > 0.15) {
      clarityScore -= 25;
    } else if (fillerRatio > 0.10) {
      clarityScore -= 15;
    } else if (fillerRatio < 0.05) {
      clarityScore += 15; // Bigger bonus for clear speech
    }
  }
 
  // Bonus for sustained speaking
  if (transcript.length > 100) {
    clarityScore = Math.min(100, clarityScore + 5);
  }
  if (transcript.length > 200) {
    clarityScore = Math.min(100, clarityScore + 5);
  }
 
  return Math.max(0, Math.min(100, clarityScore));
};
// Calculate engagement with ML-based factors
const calculateEngagement = (
  hasFace: boolean,
  transcriptLength: number,
  audioLevel: number,
  emotions: any[],
  eyeContactScore: number
): number => {
  let engagementScore = 0; // Start from 0 for accuracy
 
  // Face presence (30 points - increased)
  if (hasFace) {
    engagementScore += 30;
  }
 
  // Eye contact quality (25 points)
  engagementScore += (eyeContactScore / 100) * 25;
 
  // Speech activity (20 points)
  if (transcriptLength > 0) {
    const speechScore = Math.min(20, (transcriptLength / 500) * 20);
    engagementScore += speechScore;
  }
 
  // Audio activity (15 points)
  if (audioLevel > 0.2) {
    engagementScore += Math.min(15, audioLevel * 20);
  }
 
  // Positive emotions from ML (10 points - adjusted)
  if (emotions.length > 0) {
    const positiveEmotions = emotions.filter(e => {
      const label = e.label.toLowerCase();
      return ['happy', 'neutral'].some(p => label.includes(p)); // Updated for new model
    });
   
    for (const emotion of positiveEmotions) {
      engagementScore += emotion.score * 10;
    }
  }
 
  return Math.max(0, Math.min(100, engagementScore));
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
      facialAnalysis?.eyeContact || 0
    );
    // Return ORIGINAL structure for backward compatibility
    return {
      eyeContact: facialAnalysis?.eyeContact || 0,
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
      eyeContact: 0,
      posture: 0,
      clarity: 0,
      engagement: 0,
      bodyLanguage: 'neutral',
      emotions: [],
      imageData: null
    };
  }
};
