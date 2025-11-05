import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, Mic, MicOff, Video, VideoOff, Square, ArrowLeft, Loader2 } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { VisionAnalyzer } from "@/lib/visionAnalysis";
import { AudioAnalyzer } from "@/lib/audioAnalysis";
import { SpeechRecognitionService, SpeechAnalyzer } from "@/lib/speechRecognition";
import { FusionAlgorithm } from "@/lib/fusionAlgorithm";
import type { RawMetrics } from "@/lib/fusionAlgorithm";

const Practice = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [metrics, setMetrics] = useState({
    eyeContact: 0,
    posture: 0,
    clarity: 0,
    engagement: 0,
    pitch: 0,
    volume: 0,
    gestureVariety: 0,
    emotion: 'neutral',
  });
  const [feedback, setFeedback] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const [finalTranscript, setFinalTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [speechRecognitionSupported, setSpeechRecognitionSupported] = useState(true);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const visionAnalyzerRef = useRef<VisionAnalyzer | null>(null);
  const audioAnalyzerRef = useRef<AudioAnalyzer | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognitionService | null>(null);
  const speechAnalyzerRef = useRef<SpeechAnalyzer>(new SpeechAnalyzer());
  const fusionAlgorithmRef = useRef<FusionAlgorithm>(new FusionAlgorithm());
  const animationFrameRef = useRef<number | null>(null);
  const lastBackendAnalysisRef = useRef<number>(0);

  // Initialize advanced AI/ML models on mount
  useEffect(() => {
    const init = async () => {
      try {
        console.log("Initializing advanced AI/ML models (MediaPipe)...");
        
        // Initialize MediaPipe vision analyzer
        visionAnalyzerRef.current = new VisionAnalyzer();
        await visionAnalyzerRef.current.initialize();
        
        setModelsLoaded(true);
        console.log("MediaPipe models loaded successfully");
        
        toast({
          title: "AI Models Ready",
          description: "MediaPipe, audio analysis, and NLP ready for real-time analysis",
        });
      } catch (error) {
        console.error("Failed to initialize AI models:", error);
        setModelsLoaded(true); // Continue anyway with reduced functionality
        toast({
          title: "Model Loading Warning",
          description: "Some advanced features may be limited",
          variant: "destructive",
        });
      }
    };
    init();

    // Initialize speech recognition
    const speechService = new SpeechRecognitionService();
    if (!speechService.isSupported()) {
      setSpeechRecognitionSupported(false);
      toast({
        title: "Speech Recognition Unavailable",
        description: "Your browser doesn't support speech recognition. Try Chrome or Edge.",
        variant: "destructive",
      });
    } else {
      speechRecognitionRef.current = speechService;
      
      speechService.onTranscript((text, isFinal) => {
        if (isFinal) {
          setFinalTranscript(prev => prev + ' ' + text);
          setInterimTranscript('');
          
          // Analyze speech patterns in real-time
          speechAnalyzerRef.current.analyzeTranscript(text);
        } else {
          setInterimTranscript(text);
        }
      });

      speechService.onError((error) => {
        console.error('Speech recognition error:', error);
        if (error === 'not-allowed') {
          toast({
            title: "Microphone Permission Required",
            description: "Please allow microphone access for speech analysis",
            variant: "destructive",
          });
        }
      });
    }

    return () => {
      if (visionAnalyzerRef.current) {
        visionAnalyzerRef.current.cleanup();
      }
      if (audioAnalyzerRef.current) {
        audioAnalyzerRef.current.cleanup();
      }
      if (speechRecognitionRef.current) {
        speechRecognitionRef.current.stop();
      }
    };
  }, [toast]);

  // Real-time analysis loop with advanced AI/ML
  useEffect(() => {
    if (!isRecording || !videoRef.current || !canvasRef.current) return;

    let frameCount = 0;

    const analyzeFrame = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!video || !canvas || video.readyState !== video.HAVE_ENOUGH_DATA) {
        animationFrameRef.current = requestAnimationFrame(analyzeFrame);
        return;
      }

      frameCount++;
      const timestamp = performance.now();

      try {
        // Run MediaPipe vision analysis every frame
        const visionMetrics = visionAnalyzerRef.current 
          ? await visionAnalyzerRef.current.analyzeFrame(video, timestamp) 
          : null;
        
        // Get audio features from advanced audio analyzer
        const audioFeatures = audioAnalyzerRef.current 
          ? audioAnalyzerRef.current.getAudioFeatures() 
          : null;
        
        // Get speech analysis metrics
        const speechMetrics = speechAnalyzerRef.current.getMetrics();

        // Combine all raw metrics for fusion algorithm
        if (visionMetrics && audioFeatures) {
          const rawMetrics: RawMetrics = {
            // Vision metrics
            eyeContact: visionMetrics.face.eyeContact,
            emotion: visionMetrics.face.emotion,
            emotionConfidence: visionMetrics.face.emotionConfidence,
            postureScore: visionMetrics.posture.postureScore,
            shoulderAlignment: visionMetrics.posture.shoulderAlignment,
            headPosition: visionMetrics.posture.headPosition,
            gestureVariety: visionMetrics.gestures.gestureVariety,
            handVisibility: visionMetrics.gestures.handVisibility,
            
            // Audio metrics
            pitch: audioFeatures.pitch,
            pitchVariation: audioFeatures.pitchVariation,
            volume: audioFeatures.volume,
            volumeVariation: audioFeatures.volumeVariation,
            clarity: audioFeatures.clarity,
            energy: audioFeatures.energy,
            
            // Speech metrics
            wordsPerMinute: speechMetrics.wordsPerMinute,
            fillerCount: speechMetrics.fillerCount,
            fillerPercentage: speechMetrics.fillerPercentage,
            clarityScore: speechMetrics.clarityScore,
            fluencyScore: speechMetrics.fluencyScore,
            articulationScore: speechMetrics.articulationScore,
          };

          // Apply multi-modal fusion algorithm with temporal smoothing
          fusionAlgorithmRef.current.setContext('presentation');
          const fusedMetrics = fusionAlgorithmRef.current.fuse(rawMetrics);
          
          // Update UI metrics with fused, smoothed values
          const newMetrics = {
            eyeContact: fusedMetrics.eyeContact,
            posture: fusedMetrics.posture,
            clarity: fusedMetrics.speechClarity,
            engagement: fusedMetrics.contentEngagement,
            pitch: audioFeatures.pitch,
            volume: audioFeatures.volume,
            gestureVariety: fusedMetrics.bodyLanguage,
            emotion: visionMetrics.face.emotion,
          };
          
          setMetrics(newMetrics);

          // Generate real-time feedback based on fused metrics
          const feedbackParts = [];
          if (fusedMetrics.eyeContact < 50) feedbackParts.push("👀 Improve eye contact");
          if (fusedMetrics.posture < 60) feedbackParts.push("📏 Straighten your posture");
          if (speechMetrics.wordsPerMinute > 150) feedbackParts.push("⏱️ Slow down - speak at 120-150 WPM");
          if (speechMetrics.wordsPerMinute < 80 && speechMetrics.wordsPerMinute > 0) feedbackParts.push("⚡ Speak faster");
          if (audioFeatures.volume < -40) feedbackParts.push("🔊 Speak louder");
          if (fusedMetrics.bodyLanguage < 40) feedbackParts.push("👐 Use more hand gestures");
          if (speechMetrics.fillerPercentage > 10) feedbackParts.push(`🎯 Reduce filler words (${speechMetrics.fillerPercentage}%)`);
          if (fusedMetrics.confidence < 50) feedbackParts.push("⚠️ Low signal quality");
          
          setFeedback(feedbackParts.length > 0 ? feedbackParts.join(" • ") : "✨ Excellent! Keep it up!");
        }

        // Send to backend for deeper NLP analysis every 15 seconds
        const now = Date.now();
        if (now - lastBackendAnalysisRef.current > 15000 && finalTranscript.length > 50) {
          lastBackendAnalysisRef.current = now;
          
          const context = canvas.getContext("2d");
          if (context) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL("image/jpeg", 0.8);

            console.log('Sending to backend for deep NLP analysis...');
            const { data, error } = await supabase.functions.invoke('analyze-presentation', {
              body: {
                imageData,
                audioData: null,
                transcript: finalTranscript,
              }
            });

            if (error) {
              console.error('Backend analysis error:', error);
            } else if (data) {
              console.log('Backend deep analysis:', data);
              // Backend provides deeper semantic content analysis
            }
          }
        }
      } catch (error) {
        console.error("Error analyzing frame:", error);
      }

      animationFrameRef.current = requestAnimationFrame(analyzeFrame);
    };

    analyzeFrame();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRecording, finalTranscript]);

  // Audio level monitoring using advanced audio analyzer
  useEffect(() => {
    if (!isRecording) {
      setAudioLevel(0);
      return;
    }

    const updateAudioLevel = () => {
      if (audioAnalyzerRef.current) {
        const features = audioAnalyzerRef.current.getAudioFeatures();
        // Convert dB to 0-100 scale for visualization
        const normalizedVolume = Math.max(0, Math.min(100, (features.volume + 60) * 1.67));
        setAudioLevel(Math.round(normalizedVolume));
      }
      if (isRecording) {
        requestAnimationFrame(updateAudioLevel);
      }
    };

    updateAudioLevel();
  }, [isRecording]);

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setStream(mediaStream);
      setIsCameraOn(true);
      setIsMicOn(true);
      
      toast({
        title: "Camera Ready",
        description: "Camera and microphone are now active",
      });
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast({
        title: "Camera Access Denied",
        description: "Please allow camera and microphone access to continue",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsCameraOn(false);
      setIsMicOn(false);
    }
  };

  const toggleMicrophone = () => {
    if (stream) {
      stream.getAudioTracks().forEach(track => {
        track.enabled = !track.enabled;
      });
      setIsMicOn(!isMicOn);
    }
  };

  const startRecording = () => {
    if (!isCameraOn) {
      toast({
        title: "Camera Not Active",
        description: "Please turn on your camera first",
        variant: "destructive",
      });
      return;
    }

    if (!modelsLoaded) {
      toast({
        title: "AI Models Loading",
        description: "Please wait for AI models to initialize",
        variant: "destructive",
      });
      return;
    }

    if (!stream) return;
    
    setIsRecording(true);
    setRecordingTime(0);
    setFinalTranscript("");
    setInterimTranscript("");
    setFeedback("");
    speechAnalyzerRef.current.reset();
    fusionAlgorithmRef.current.reset();
    lastBackendAnalysisRef.current = 0;

    // Initialize advanced audio analyzer with the stream
    audioAnalyzerRef.current = new AudioAnalyzer();
    audioAnalyzerRef.current.initialize(stream);
    
    // Start speech recognition
    if (speechRecognitionRef.current && speechRecognitionSupported) {
      const started = speechRecognitionRef.current.start();
      if (!started) {
        toast({
          title: "Speech Recognition Failed",
          description: "Could not start speech recognition",
          variant: "destructive",
        });
      }
    }
    
    toast({
      title: "Recording Started",
      description: "Real-time AI/ML analysis active with MediaPipe, audio analysis, and NLP",
    });
  };

  const stopRecording = () => {
    setIsRecording(false);
    
    // Stop speech recognition
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.stop();
    }

    // Cleanup audio analyzer
    if (audioAnalyzerRef.current) {
      audioAnalyzerRef.current.cleanup();
    }

    // Stop animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const finalAnalysis = speechAnalyzerRef.current.getMetrics();
    
    toast({
      title: "Recording Stopped",
      description: "Analyzing your complete session...",
    });
    
    // Navigate to results with complete data
    setTimeout(() => {
      navigate("/results", { 
        state: { 
          duration: recordingTime,
          metrics: metrics,
          transcript: finalTranscript,
          speechAnalysis: finalAnalysis,
          feedback: feedback,
        } 
      });
    }, 2000);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-gradient-hero p-4">
      <div className="container mx-auto max-w-7xl">
        <div className="flex items-center gap-4 mb-6">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/")}
            className="text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Video Area */}
          <div className="lg:col-span-2">
            <Card className="p-6 bg-gradient-card border-border">
              <div className="relative aspect-video bg-background rounded-lg overflow-hidden mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!isCameraOn && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background/90">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-muted-foreground">Camera is off</p>
                    </div>
                  </div>
                )}

                {isRecording && (
                  <>
                    <div className="absolute top-4 left-4 flex flex-col gap-2">
                      <div className="flex items-center gap-2 px-3 py-2 bg-primary/90 rounded-full">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                        <span className="text-sm font-medium text-white">Live AI Analysis</span>
                      </div>
                      {(finalTranscript || interimTranscript) && (
                        <div className="px-3 py-2 bg-background/90 rounded-lg max-w-md max-h-32 overflow-y-auto">
                          <p className="text-xs text-foreground">
                            {finalTranscript} <span className="text-muted-foreground italic">{interimTranscript}</span>
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="absolute top-4 right-4 px-4 py-2 bg-background/90 rounded-full">
                      <span className="text-lg font-bold text-primary">{formatTime(recordingTime)}</span>
                    </div>

                    {feedback && (
                      <div className="absolute bottom-4 left-4 right-4">
                        <div className="px-4 py-3 bg-primary/90 rounded-lg">
                          <p className="text-sm font-medium text-white">{feedback}</p>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Controls */}
              <div className="flex justify-center gap-4">
                {!isCameraOn ? (
                  <Button
                    size="lg"
                    onClick={startCamera}
                    className="bg-primary hover:bg-primary/90"
                  >
                    <Camera className="w-5 h-5 mr-2" />
                    Turn On Camera
                  </Button>
                ) : (
                  <>
                    <Button
                      size="lg"
                      variant="outline"
                      onClick={stopCamera}
                      className="border-border"
                    >
                      <VideoOff className="w-5 h-5 mr-2" />
                      Stop Camera
                    </Button>
                    
                    <Button
                      size="lg"
                      variant="outline"
                      onClick={toggleMicrophone}
                      className="border-border"
                    >
                      {isMicOn ? (
                        <>
                          <Mic className="w-5 h-5 mr-2" />
                          Mic On
                        </>
                      ) : (
                        <>
                          <MicOff className="w-5 h-5 mr-2" />
                          Mic Off
                        </>
                      )}
                    </Button>

                    {!isRecording ? (
                      <Button
                        size="lg"
                        onClick={startRecording}
                        className="bg-primary hover:bg-primary/90"
                        disabled={!modelsLoaded}
                      >
                        <Video className="w-5 h-5 mr-2" />
                        {modelsLoaded ? "Start Recording" : "Loading AI..."}
                      </Button>
                    ) : (
                      <Button
                        size="lg"
                        onClick={stopRecording}
                        variant="destructive"
                      >
                        <Square className="w-5 h-5 mr-2" />
                        Stop Recording
                      </Button>
                    )}
                  </>
                )}
              </div>
            </Card>
          </div>

          {/* Real-Time Analytics */}
          <div className="space-y-4">
            <Card className="p-6 bg-gradient-card border-border">
              <h3 className="text-lg font-bold mb-4 text-foreground">Real-Time AI Analysis</h3>
              
              {!isRecording ? (
                <div className="text-center py-8">
                  <Loader2 className="w-12 h-12 mx-auto mb-3 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    {modelsLoaded ? "Start recording to see live AI analysis" : "Loading AI models..."}
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Eye Contact */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Eye Contact</span>
                      <span className="text-sm font-bold text-primary">{metrics.eyeContact}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-primary transition-all duration-500"
                        style={{ width: `${metrics.eyeContact}%` }}
                      />
                    </div>
                  </div>

                  {/* Posture */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Posture</span>
                      <span className="text-sm font-bold text-primary">{metrics.posture}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-secondary transition-all duration-500"
                        style={{ width: `${metrics.posture}%` }}
                      />
                    </div>
                  </div>

                  {/* Clarity */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Clarity</span>
                      <span className="text-sm font-bold text-primary">{metrics.clarity}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-accent transition-all duration-500"
                        style={{ width: `${metrics.clarity}%` }}
                      />
                    </div>
                  </div>

                  {/* Engagement */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Engagement</span>
                      <span className="text-sm font-bold text-primary">{metrics.engagement}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-primary transition-all duration-500"
                        style={{ width: `${metrics.engagement}%` }}
                      />
                    </div>
                  </div>

                  {/* Audio Level */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Audio Level</span>
                      <span className="text-sm font-bold text-primary">{audioLevel}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500 transition-all duration-100"
                        style={{ width: `${audioLevel}%` }}
                      />
                    </div>
                  </div>

                  {/* Additional Metrics */}
                  <div className="grid grid-cols-2 gap-3 pt-4 border-t border-border">
                    <div className="text-center">
                      <p className="text-xs text-muted-foreground mb-1">Emotion</p>
                      <p className="text-sm font-bold text-foreground capitalize">{metrics.emotion}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-muted-foreground mb-1">Gestures</p>
                      <p className="text-sm font-bold text-foreground">{metrics.gestureVariety}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-muted-foreground mb-1">Pitch</p>
                      <p className="text-sm font-bold text-foreground">{metrics.pitch} Hz</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-muted-foreground mb-1">Volume</p>
                      <p className="text-sm font-bold text-foreground">{metrics.volume} dB</p>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Practice;
