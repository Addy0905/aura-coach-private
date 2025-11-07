import { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, Mic, MicOff, Video, VideoOff, Square, ArrowLeft, Loader2 } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { VisionAnalyzer } from "@/lib/visionAnalysis";
import { AudioAnalyzer } from "@/lib/audioAnalysis";
import { SpeechRecognitionService, SpeechAnalyzer } from "@/lib/speechRecognition";
import { ContentAnalyzer } from "@/lib/contentAnalysis";
import { FusionAlgorithm } from "@/lib/fusionAlgorithm";
import type { RawMetrics } from "@/lib/fusionAlgorithm";
import CategorySelection, { type UserCategory } from "@/components/CategorySelection";

const Practice = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  const [selectedCategory, setSelectedCategory] = useState<UserCategory | null>(
    (location.state?.category as UserCategory) || null
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
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
  const contentAnalyzerRef = useRef<ContentAnalyzer>(new ContentAnalyzer());
  const fusionAlgorithmRef = useRef<FusionAlgorithm>(new FusionAlgorithm());
  const animationFrameRef = useRef<number | null>(null);
  const lastBackendAnalysisRef = useRef<number>(0);

  // Initialize advanced AI/ML models on mount
  useEffect(() => {
    const init = async () => {
      try {
        console.log("Initializing advanced AI/ML models (MediaPipe)...");
       
        visionAnalyzerRef.current = new VisionAnalyzer();
        await visionAnalyzerRef.current.initialize();
       
        setModelsLoaded(true);
        console.log("MediaPipe models loaded successfully");
       
        toast({
          title: "AI Models Ready",
          description: "MediaPipe Face Mesh (468 landmarks), Pose (33 keypoints), YIN pitch detection, SNR clarity, TF-IDF, and sentiment analysis ready",
        });
      } catch (error) {
        console.error("Failed to initialize AI models:", error);
        setModelsLoaded(true);
        toast({
          title: "Model Loading Warning",
          description: "Some advanced features may be limited",
          variant: "destructive",
        });
      }
    };
    init();

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
      if (visionAnalyzerRef.current) visionAnalyzerRef.current.cleanup();
      if (audioAnalyzerRef.current) audioAnalyzerRef.current.cleanup();
      if (speechRecognitionRef.current) speechRecognitionRef.current.stop();
    };
  }, [toast]);

  // Real-time analysis loop
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
        const visionMetrics = visionAnalyzerRef.current
          ? await visionAnalyzerRef.current.analyzeFrame(video, timestamp)
          : null;
       
        const overlayCanvas = overlayCanvasRef.current;
        if (overlayCanvas && visionMetrics) {
          overlayCanvas.width = video.videoWidth;
          overlayCanvas.height = video.videoHeight;
          const ctx = overlayCanvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
           
            if (visionMetrics.face.landmarks?.length) {
              ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
              visionMetrics.face.landmarks.forEach((landmark: any) => {
                const x = landmark.x * overlayCanvas.width;
                const y = landmark.y * overlayCanvas.height;
                ctx.beginPath();
                ctx.arc(x, y, 1, 0, 2 * Math.PI);
                ctx.fill();
              });
            }
           
            if (visionMetrics.posture.landmarks?.length) {
              ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
              ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
              ctx.lineWidth = 2;
             
              visionMetrics.posture.landmarks.forEach((landmark: any) => {
                const x = landmark.x * overlayCanvas.width;
                const y = landmark.y * overlayCanvas.height;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
              });
             
              const connections = [
                [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
                [11, 23], [12, 24], [23, 24],
                [23, 25], [25, 27], [24, 26], [26, 28],
                [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8]
              ];
             
              connections.forEach(([start, end]) => {
                const l = visionMetrics.posture.landmarks;
                if (l[start] && l[end]) {
                  const sx = l[start].x * overlayCanvas.width;
                  const sy = l[start].y * overlayCanvas.height;
                  const ex = l[end].x * overlayCanvas.width;
                  const ey = l[end].y * overlayCanvas.height;
                  ctx.beginPath();
                  ctx.moveTo(sx, sy);
                  ctx.lineTo(ex, ey);
                  ctx.stroke();
                }
              });
            }
          }
        }
       
        const audioFeatures = audioAnalyzerRef.current?.getAudioFeatures() || null;
        const speechMetrics = speechAnalyzerRef.current.getMetrics();
        const contentMetrics = finalTranscript.length > 20
          ? contentAnalyzerRef.current.analyzeContent(finalTranscript)
          : null;

        if (visionMetrics && audioFeatures) {
          const rawMetrics: RawMetrics = {
            eyeContact: visionMetrics.face.eyeContact,
            emotion: visionMetrics.face.emotion,
            emotionConfidence: visionMetrics.face.emotionConfidence,
            postureScore: visionMetrics.posture.postureScore,
            shoulderAlignment: visionMetrics.posture.shoulderAlignment,
            headPosition: visionMetrics.posture.headPosition,
            gestureVariety: visionMetrics.gestures.gestureVariety,
            handVisibility: visionMetrics.gestures.handVisibility,
           
            pitch: audioFeatures.pitch,
            pitchVariation: audioFeatures.pitchVariation,
            volume: audioFeatures.volume,
            volumeVariation: audioFeatures.volumeVariation,
            clarity: audioFeatures.clarity,
            energy: audioFeatures.energy,
           
            wordsPerMinute: speechMetrics.wordsPerMinute,
            fillerCount: speechMetrics.fillerCount,
            fillerPercentage: speechMetrics.fillerPercentage,
            clarityScore: speechMetrics.clarityScore,
            fluencyScore: speechMetrics.fluencyScore,
            articulationScore: speechMetrics.articulationScore,
          };

          fusionAlgorithmRef.current.setContext(selectedCategory || 'students');
          const fusedMetrics = fusionAlgorithmRef.current.fuse(rawMetrics);
         
          const contentBoost = contentMetrics ? (contentMetrics.coherenceScore / 100) * 10 : 0;
          const newMetrics = {
            eyeContact: fusedMetrics.eyeContact,
            posture: fusedMetrics.posture,
            clarity: fusedMetrics.speechClarity, // ← Real 0–100 clarity
            engagement: Math.min(100, fusedMetrics.contentEngagement + contentBoost),
            pitch: audioFeatures.pitch,
            volume: audioFeatures.volume,
            gestureVariety: fusedMetrics.bodyLanguage,
            emotion: visionMetrics.face.emotion,
          };
         
          setMetrics(newMetrics);

          const feedbackParts = [];
          if (fusedMetrics.eyeContact < 50) feedbackParts.push("Improve eye contact");
          if (fusedMetrics.posture < 60) feedbackParts.push("Straighten your posture");
          if (speechMetrics.wordsPerMinute > 150) feedbackParts.push("Slow down - speak at 120-150 WPM");
          if (speechMetrics.wordsPerMinute < 80 && speechMetrics.wordsPerMinute > 0) feedbackParts.push("Speak faster");
          if (audioFeatures.volume < -40) feedbackParts.push("Speak louder");
          if (fusedMetrics.bodyLanguage < 40) feedbackParts.push("Use more hand gestures");
          if (speechMetrics.fillerPercentage > 10) feedbackParts.push(`Reduce filler words (${speechMetrics.fillerPercentage}%)`);
          if (contentMetrics && contentMetrics.coherenceScore < 60) feedbackParts.push("Improve flow and coherence");
          if (contentMetrics && contentMetrics.sentimentLabel === 'negative') feedbackParts.push("Use more positive language");
          if (fusedMetrics.confidence < 50) feedbackParts.push("Low signal quality");
         
          setFeedback(feedbackParts.length > 0 ? feedbackParts.join(" • ") : "Excellent! Keep it up!");
        }

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
              body: { imageData, audioData: null, transcript: finalTranscript }
            });
            if (error) console.error('Backend analysis error:', error);
            else if (data) console.log('Backend deep analysis:', data);
          }
        }
      } catch (error) {
        console.error("Error analyzing frame:", error);
      }

      animationFrameRef.current = requestAnimationFrame(analyzeFrame);
    };

    analyzeFrame();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isRecording, finalTranscript]);

  // Audio level monitoring
  useEffect(() => {
    if (!isRecording) {
      setAudioLevel(0);
      return;
    }

    const updateAudioLevel = () => {
      if (audioAnalyzerRef.current) {
        const features = audioAnalyzerRef.current.getAudioFeatures();
        const normalizedVolume = Math.max(0, Math.min(100, (features.volume + 60) * 1.67));
        setAudioLevel(Math.round(normalizedVolume));
      }
      if (isRecording) requestAnimationFrame(updateAudioLevel);
    };
    updateAudioLevel();
  }, [isRecording]);

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => setRecordingTime(prev => prev + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      });
     
      if (videoRef.current) videoRef.current.srcObject = mediaStream;
     
      setStream(mediaStream);
      setIsCameraOn(true);
      setIsMicOn(true);
     
      toast({ title: "Camera Ready", description: "Camera and microphone are now active" });
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
      stream.getAudioTracks().forEach(track => { track.enabled = !track.enabled; });
      setIsMicOn(prev => !prev);
    }
  };

  const startRecording = () => {
    if (!isCameraOn) {
      toast({ title: "Camera Not Active", description: "Please turn on your camera first", variant: "destructive" });
      return;
    }
    if (!modelsLoaded) {
      toast({ title: "AI Models Loading", description: "Please wait for AI models to initialize", variant: "destructive" });
      return;
    }
    if (!stream) return;
   
    setIsRecording(true);
    setRecordingTime(0);
    setFinalTranscript("");
    setInterimTranscript("");
    setFeedback("");
    speechAnalyzerRef.current.reset();
    contentAnalyzerRef.current.reset();
    fusionAlgorithmRef.current.reset();
    lastBackendAnalysisRef.current = 0;

    audioAnalyzerRef.current = new AudioAnalyzer();
    audioAnalyzerRef.current.initialize(stream);
   
    if (speechRecognitionRef.current && speechRecognitionSupported) {
      const started = speechRecognitionRef.current.start();
      if (!started) {
        toast({ title: "Speech Recognition Failed", description: "Could not start speech recognition", variant: "destructive" });
      }
    }
   
    toast({
      title: "Recording Started",
      description: "Real-time analysis: MediaPipe (face/pose), YIN pitch, SNR clarity, TF-IDF keywords, sentiment, and NER active",
    });
  };

  const stopRecording = () => {
    setIsRecording(false);
   
    if (speechRecognitionRef.current) speechRecognitionRef.current.stop();
    if (audioAnalyzerRef.current) audioAnalyzerRef.current.cleanup();
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    const finalAnalysis = speechAnalyzerRef.current.getMetrics();
   
    toast({ title: "Recording Stopped", description: "Analyzing your complete session..." });
   
    setTimeout(() => {
      const contentMetrics = finalTranscript.length > 20
        ? contentAnalyzerRef.current.analyzeContent(finalTranscript)
        : null;
     
      navigate("/results", {
        state: {
          duration: recordingTime,
          metrics,
          transcript: finalTranscript,
          speechAnalysis: finalAnalysis,
          contentAnalysis: contentMetrics,
          feedback,
          category: selectedCategory,
        }
      });
    }, 2000);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (!selectedCategory) {
    return <CategorySelection onSelect={setSelectedCategory} />;
  }

  return (
    <div className="min-h-screen bg-gradient-hero p-4">
      <div className="container mx-auto max-w-7xl">
        <div className="flex items-center gap-4 mb-6">
          <Button variant="ghost" size="sm" onClick={() => navigate("/")} className="text-muted-foreground hover:text-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Home
          </Button>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Video Area */}
          <div className="lg:col-span-2">
            <Card className="p-6 bg-gradient-card border-border">
              <div className="relative aspect-video bg-background rounded-lg overflow-hidden mb-4">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                <canvas ref={canvasRef} className="hidden" />
                <canvas ref={overlayCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" style={{ objectFit: 'cover' }} />
               
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

              <div className="flex justify-center gap-4">
                {!isCameraOn ? (
                  <Button size="lg" onClick={startCamera} className="bg-primary hover:bg-primary/90">
                    <Camera className="w-5 h-5 mr-2" /> Turn On Camera
                  </Button>
                ) : (
                  <>
                    <Button size="lg" variant="outline" onClick={stopCamera} className="border-border">
                      <VideoOff className="w-5 h-5 mr-2" /> Stop Camera
                    </Button>
                   
                    <Button size="lg" variant="outline" onClick={toggleMicrophone} className="border-border">
                      {isMicOn ? (
                        <> <Mic className="w-5 h-5 mr-2" /> Mic On </>
                      ) : (
                        <> <MicOff className="w-5 h-5 mr-2" /> Mic Off </>
                      )}
                    </Button>

                    {!isRecording ? (
                      <Button size="lg" onClick={startRecording} className="bg-primary hover:bg-primary/90" disabled={!modelsLoaded}>
                        <Video className="w-5 h-5 mr-2" />
                        {modelsLoaded ? "Start Recording" : "Loading AI..."}
                      </Button>
                    ) : (
                      <Button size="lg" onClick={stopRecording} variant="destructive">
                        <Square className="w-5 h-5 mr-2" /> Stop Recording
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
                      <div className="h-full bg-gradient-primary transition-all duration-500" style={{ width: `${metrics.eyeContact}%` }} />
                    </div>
                  </div>

                  {/* Posture */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Posture</span>
                      <span className="text-sm font-bold text-primary">{metrics.posture}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-secondary transition-all duration-500" style={{ width: `${metrics.posture}%` }} />
                    </div>
                  </div>

                  {/* Clarity — NOW STARTS AT 0% */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Clarity</span>
                      <span className="text-sm font-bold text-primary">{metrics.clarity}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-accent transition-all duration-500"
                        style={{ width: `${Math.max(0, metrics.clarity)}%` }} // ← Ensures 0% when silent
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
                      <div className="h-full bg-gradient-primary transition-all duration-500" style={{ width: `${metrics.engagement}%` }} />
                    </div>
                  </div>

                  {/* Audio Level */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Audio Level</span>
                      <span className="text-sm font-bold text-primary">{audioLevel}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 transition-all duration-100" style={{ width: `${audioLevel}%` }} />
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

export default Practice;import { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, Mic, MicOff, Video, VideoOff, Square, ArrowLeft, Loader2 } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { VisionAnalyzer } from "@/lib/visionAnalysis";
import { AudioAnalyzer } from "@/lib/audioAnalysis";
import { SpeechRecognitionService, SpeechAnalyzer } from "@/lib/speechRecognition";
import { ContentAnalyzer } from "@/lib/contentAnalysis";
import { FusionAlgorithm } from "@/lib/fusionAlgorithm";
import type { RawMetrics } from "@/lib/fusionAlgorithm";
import CategorySelection, { type UserCategory } from "@/components/CategorySelection";

const Practice = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  const [selectedCategory, setSelectedCategory] = useState<UserCategory | null>(
    (location.state?.category as UserCategory) || null
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
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
  const contentAnalyzerRef = useRef<ContentAnalyzer>(new ContentAnalyzer());
  const fusionAlgorithmRef = useRef<FusionAlgorithm>(new FusionAlgorithm());
  const animationFrameRef = useRef<number | null>(null);
  const lastBackendAnalysisRef = useRef<number>(0);

  // Initialize advanced AI/ML models on mount
  useEffect(() => {
    const init = async () => {
      try {
        console.log("Initializing advanced AI/ML models (MediaPipe)...");
       
        visionAnalyzerRef.current = new VisionAnalyzer();
        await visionAnalyzerRef.current.initialize();
       
        setModelsLoaded(true);
        console.log("MediaPipe models loaded successfully");
       
        toast({
          title: "AI Models Ready",
          description: "MediaPipe Face Mesh (468 landmarks), Pose (33 keypoints), YIN pitch detection, SNR clarity, TF-IDF, and sentiment analysis ready",
        });
      } catch (error) {
        console.error("Failed to initialize AI models:", error);
        setModelsLoaded(true);
        toast({
          title: "Model Loading Warning",
          description: "Some advanced features may be limited",
          variant: "destructive",
        });
      }
    };
    init();

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
      if (visionAnalyzerRef.current) visionAnalyzerRef.current.cleanup();
      if (audioAnalyzerRef.current) audioAnalyzerRef.current.cleanup();
      if (speechRecognitionRef.current) speechRecognitionRef.current.stop();
    };
  }, [toast]);

  // Real-time analysis loop
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
        const visionMetrics = visionAnalyzerRef.current
          ? await visionAnalyzerRef.current.analyzeFrame(video, timestamp)
          : null;
       
        const overlayCanvas = overlayCanvasRef.current;
        if (overlayCanvas && visionMetrics) {
          overlayCanvas.width = video.videoWidth;
          overlayCanvas.height = video.videoHeight;
          const ctx = overlayCanvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
           
            if (visionMetrics.face.landmarks?.length) {
              ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
              visionMetrics.face.landmarks.forEach((landmark: any) => {
                const x = landmark.x * overlayCanvas.width;
                const y = landmark.y * overlayCanvas.height;
                ctx.beginPath();
                ctx.arc(x, y, 1, 0, 2 * Math.PI);
                ctx.fill();
              });
            }
           
            if (visionMetrics.posture.landmarks?.length) {
              ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
              ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
              ctx.lineWidth = 2;
             
              visionMetrics.posture.landmarks.forEach((landmark: any) => {
                const x = landmark.x * overlayCanvas.width;
                const y = landmark.y * overlayCanvas.height;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
              });
             
              const connections = [
                [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
                [11, 23], [12, 24], [23, 24],
                [23, 25], [25, 27], [24, 26], [26, 28],
                [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8]
              ];
             
              connections.forEach(([start, end]) => {
                const l = visionMetrics.posture.landmarks;
                if (l[start] && l[end]) {
                  const sx = l[start].x * overlayCanvas.width;
                  const sy = l[start].y * overlayCanvas.height;
                  const ex = l[end].x * overlayCanvas.width;
                  const ey = l[end].y * overlayCanvas.height;
                  ctx.beginPath();
                  ctx.moveTo(sx, sy);
                  ctx.lineTo(ex, ey);
                  ctx.stroke();
                }
              });
            }
          }
        }
       
        const audioFeatures = audioAnalyzerRef.current?.getAudioFeatures() || null;
        const speechMetrics = speechAnalyzerRef.current.getMetrics();
        const contentMetrics = finalTranscript.length > 20
          ? contentAnalyzerRef.current.analyzeContent(finalTranscript)
          : null;

        if (visionMetrics && audioFeatures) {
          const rawMetrics: RawMetrics = {
            eyeContact: visionMetrics.face.eyeContact,
            emotion: visionMetrics.face.emotion,
            emotionConfidence: visionMetrics.face.emotionConfidence,
            postureScore: visionMetrics.posture.postureScore,
            shoulderAlignment: visionMetrics.posture.shoulderAlignment,
            headPosition: visionMetrics.posture.headPosition,
            gestureVariety: visionMetrics.gestures.gestureVariety,
            handVisibility: visionMetrics.gestures.handVisibility,
           
            pitch: audioFeatures.pitch,
            pitchVariation: audioFeatures.pitchVariation,
            volume: audioFeatures.volume,
            volumeVariation: audioFeatures.volumeVariation,
            clarity: audioFeatures.clarity,
            energy: audioFeatures.energy,
           
            wordsPerMinute: speechMetrics.wordsPerMinute,
            fillerCount: speechMetrics.fillerCount,
            fillerPercentage: speechMetrics.fillerPercentage,
            clarityScore: speechMetrics.clarityScore,
            fluencyScore: speechMetrics.fluencyScore,
            articulationScore: speechMetrics.articulationScore,
          };

          fusionAlgorithmRef.current.setContext(selectedCategory || 'students');
          const fusedMetrics = fusionAlgorithmRef.current.fuse(rawMetrics);
         
          const contentBoost = contentMetrics ? (contentMetrics.coherenceScore / 100) * 10 : 0;
          const newMetrics = {
            eyeContact: fusedMetrics.eyeContact,
            posture: fusedMetrics.posture,
            clarity: fusedMetrics.speechClarity, // ← Real 0–100 clarity
            engagement: Math.min(100, fusedMetrics.contentEngagement + contentBoost),
            pitch: audioFeatures.pitch,
            volume: audioFeatures.volume,
            gestureVariety: fusedMetrics.bodyLanguage,
            emotion: visionMetrics.face.emotion,
          };
         
          setMetrics(newMetrics);

          const feedbackParts = [];
          if (fusedMetrics.eyeContact < 50) feedbackParts.push("Improve eye contact");
          if (fusedMetrics.posture < 60) feedbackParts.push("Straighten your posture");
          if (speechMetrics.wordsPerMinute > 150) feedbackParts.push("Slow down - speak at 120-150 WPM");
          if (speechMetrics.wordsPerMinute < 80 && speechMetrics.wordsPerMinute > 0) feedbackParts.push("Speak faster");
          if (audioFeatures.volume < -40) feedbackParts.push("Speak louder");
          if (fusedMetrics.bodyLanguage < 40) feedbackParts.push("Use more hand gestures");
          if (speechMetrics.fillerPercentage > 10) feedbackParts.push(`Reduce filler words (${speechMetrics.fillerPercentage}%)`);
          if (contentMetrics && contentMetrics.coherenceScore < 60) feedbackParts.push("Improve flow and coherence");
          if (contentMetrics && contentMetrics.sentimentLabel === 'negative') feedbackParts.push("Use more positive language");
          if (fusedMetrics.confidence < 50) feedbackParts.push("Low signal quality");
         
          setFeedback(feedbackParts.length > 0 ? feedbackParts.join(" • ") : "Excellent! Keep it up!");
        }

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
              body: { imageData, audioData: null, transcript: finalTranscript }
            });
            if (error) console.error('Backend analysis error:', error);
            else if (data) console.log('Backend deep analysis:', data);
          }
        }
      } catch (error) {
        console.error("Error analyzing frame:", error);
      }

      animationFrameRef.current = requestAnimationFrame(analyzeFrame);
    };

    analyzeFrame();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isRecording, finalTranscript]);

  // Audio level monitoring
  useEffect(() => {
    if (!isRecording) {
      setAudioLevel(0);
      return;
    }

    const updateAudioLevel = () => {
      if (audioAnalyzerRef.current) {
        const features = audioAnalyzerRef.current.getAudioFeatures();
        const normalizedVolume = Math.max(0, Math.min(100, (features.volume + 60) * 1.67));
        setAudioLevel(Math.round(normalizedVolume));
      }
      if (isRecording) requestAnimationFrame(updateAudioLevel);
    };
    updateAudioLevel();
  }, [isRecording]);

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => setRecordingTime(prev => prev + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      });
     
      if (videoRef.current) videoRef.current.srcObject = mediaStream;
     
      setStream(mediaStream);
      setIsCameraOn(true);
      setIsMicOn(true);
     
      toast({ title: "Camera Ready", description: "Camera and microphone are now active" });
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
      stream.getAudioTracks().forEach(track => { track.enabled = !track.enabled; });
      setIsMicOn(prev => !prev);
    }
  };

  const startRecording = () => {
    if (!isCameraOn) {
      toast({ title: "Camera Not Active", description: "Please turn on your camera first", variant: "destructive" });
      return;
    }
    if (!modelsLoaded) {
      toast({ title: "AI Models Loading", description: "Please wait for AI models to initialize", variant: "destructive" });
      return;
    }
    if (!stream) return;
   
    setIsRecording(true);
    setRecordingTime(0);
    setFinalTranscript("");
    setInterimTranscript("");
    setFeedback("");
    speechAnalyzerRef.current.reset();
    contentAnalyzerRef.current.reset();
    fusionAlgorithmRef.current.reset();
    lastBackendAnalysisRef.current = 0;

    audioAnalyzerRef.current = new AudioAnalyzer();
    audioAnalyzerRef.current.initialize(stream);
   
    if (speechRecognitionRef.current && speechRecognitionSupported) {
      const started = speechRecognitionRef.current.start();
      if (!started) {
        toast({ title: "Speech Recognition Failed", description: "Could not start speech recognition", variant: "destructive" });
      }
    }
   
    toast({
      title: "Recording Started",
      description: "Real-time analysis: MediaPipe (face/pose), YIN pitch, SNR clarity, TF-IDF keywords, sentiment, and NER active",
    });
  };

  const stopRecording = () => {
    setIsRecording(false);
   
    if (speechRecognitionRef.current) speechRecognitionRef.current.stop();
    if (audioAnalyzerRef.current) audioAnalyzerRef.current.cleanup();
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    const finalAnalysis = speechAnalyzerRef.current.getMetrics();
   
    toast({ title: "Recording Stopped", description: "Analyzing your complete session..." });
   
    setTimeout(() => {
      const contentMetrics = finalTranscript.length > 20
        ? contentAnalyzerRef.current.analyzeContent(finalTranscript)
        : null;
     
      navigate("/results", {
        state: {
          duration: recordingTime,
          metrics,
          transcript: finalTranscript,
          speechAnalysis: finalAnalysis,
          contentAnalysis: contentMetrics,
          feedback,
          category: selectedCategory,
        }
      });
    }, 2000);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (!selectedCategory) {
    return <CategorySelection onSelect={setSelectedCategory} />;
  }

  return (
    <div className="min-h-screen bg-gradient-hero p-4">
      <div className="container mx-auto max-w-7xl">
        <div className="flex items-center gap-4 mb-6">
          <Button variant="ghost" size="sm" onClick={() => navigate("/")} className="text-muted-foreground hover:text-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Home
          </Button>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Video Area */}
          <div className="lg:col-span-2">
            <Card className="p-6 bg-gradient-card border-border">
              <div className="relative aspect-video bg-background rounded-lg overflow-hidden mb-4">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                <canvas ref={canvasRef} className="hidden" />
                <canvas ref={overlayCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" style={{ objectFit: 'cover' }} />
               
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

              <div className="flex justify-center gap-4">
                {!isCameraOn ? (
                  <Button size="lg" onClick={startCamera} className="bg-primary hover:bg-primary/90">
                    <Camera className="w-5 h-5 mr-2" /> Turn On Camera
                  </Button>
                ) : (
                  <>
                    <Button size="lg" variant="outline" onClick={stopCamera} className="border-border">
                      <VideoOff className="w-5 h-5 mr-2" /> Stop Camera
                    </Button>
                   
                    <Button size="lg" variant="outline" onClick={toggleMicrophone} className="border-border">
                      {isMicOn ? (
                        <> <Mic className="w-5 h-5 mr-2" /> Mic On </>
                      ) : (
                        <> <MicOff className="w-5 h-5 mr-2" /> Mic Off </>
                      )}
                    </Button>

                    {!isRecording ? (
                      <Button size="lg" onClick={startRecording} className="bg-primary hover:bg-primary/90" disabled={!modelsLoaded}>
                        <Video className="w-5 h-5 mr-2" />
                        {modelsLoaded ? "Start Recording" : "Loading AI..."}
                      </Button>
                    ) : (
                      <Button size="lg" onClick={stopRecording} variant="destructive">
                        <Square className="w-5 h-5 mr-2" /> Stop Recording
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
                      <div className="h-full bg-gradient-primary transition-all duration-500" style={{ width: `${metrics.eyeContact}%` }} />
                    </div>
                  </div>

                  {/* Posture */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Posture</span>
                      <span className="text-sm font-bold text-primary">{metrics.posture}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-secondary transition-all duration-500" style={{ width: `${metrics.posture}%` }} />
                    </div>
                  </div>

                  {/* Clarity — NOW STARTS AT 0% */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Clarity</span>
                      <span className="text-sm font-bold text-primary">{metrics.clarity}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-accent transition-all duration-500"
                        style={{ width: `${Math.max(0, metrics.clarity)}%` }} // ← Ensures 0% when silent
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
                      <div className="h-full bg-gradient-primary transition-all duration-500" style={{ width: `${metrics.engagement}%` }} />
                    </div>
                  </div>

                  {/* Audio Level */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Audio Level</span>
                      <span className="text-sm font-bold text-primary">{audioLevel}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 transition-all duration-100" style={{ width: `${audioLevel}%` }} />
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
