import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, Mic, MicOff, Video, VideoOff, Square, ArrowLeft, Loader2 } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { initializeModels, generateMetrics } from "@/lib/aiAnalysis";
import { SpeechRecognitionService, SpeechAnalyzer } from "@/lib/speechRecognition";

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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState({
    eyeContact: 0,
    posture: 0,
    clarity: 0,
    engagement: 0,
  });
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [speechRecognitionSupported, setSpeechRecognitionSupported] = useState(true);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyzerRef = useRef<AnalyserNode | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognitionService | null>(null);
  const speechAnalyzerRef = useRef<SpeechAnalyzer>(new SpeechAnalyzer());

  // Initialize AI models and speech recognition on mount
  useEffect(() => {
    // Initialize visual AI models
    initializeModels().then((success) => {
      setModelsLoaded(success);
      if (success) {
        console.log("Visual AI models initialized");
      }
    });

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
          setTranscript(prev => prev + ' ' + text);
          setInterimTranscript('');
          
          // Analyze speech patterns
          const analysis = speechAnalyzerRef.current.analyzeTranscript(text);
          console.log('Speech analysis:', analysis);
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

      toast({
        title: "System Ready",
        description: "AI analysis and speech recognition initialized",
      });
    }

    return () => {
      if (speechRecognitionRef.current) {
        speechRecognitionRef.current.stop();
      }
    };
  }, [toast]);

  // Real-time AI analysis during recording
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording && videoRef.current && modelsLoaded) {
      interval = setInterval(async () => {
        setRecordingTime(prev => prev + 1);
        
        // Get speech analysis metrics
        const speechMetrics = speechAnalyzerRef.current.analyzeTranscript(transcript);
        
        // Generate visual AI-powered metrics
        try {
          const visualMetrics = await generateMetrics(
            videoRef.current!,
            audioLevel,
            transcript
          );
          
          // Combine visual and speech metrics
          setLiveMetrics({
            eyeContact: visualMetrics.eyeContact,
            posture: visualMetrics.posture,
            clarity: speechMetrics.clarityScore,
            engagement: Math.round((visualMetrics.engagement + speechMetrics.fluencyScore) / 2),
          });

          // Send to backend for deeper analysis every 10 seconds
          if (recordingTime % 10 === 0 && visualMetrics.imageData && transcript.length > 20) {
            const { data, error } = await supabase.functions.invoke('analyze-presentation', {
              body: {
                imageData: visualMetrics.imageData,
                transcript: transcript
              }
            });

            if (!error && data) {
              console.log('Deep AI analysis:', data);
            }
          }
        } catch (error) {
          console.error('Error generating metrics:', error);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording, modelsLoaded, audioLevel, transcript, recordingTime]);

  // Audio level monitoring
  useEffect(() => {
    if (stream && isRecording) {
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyzer = audioContext.createAnalyser();
      analyzer.fftSize = 256;
      source.connect(analyzer);
      
      audioContextRef.current = audioContext;
      analyzerRef.current = analyzer;

      const dataArray = new Uint8Array(analyzer.frequencyBinCount);
      const updateAudioLevel = () => {
        if (analyzerRef.current && isRecording) {
          analyzerRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setAudioLevel(average / 255);
          requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();
    }

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [stream, isRecording]);

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

    if (!speechRecognitionSupported) {
      toast({
        title: "Speech Recognition Required",
        description: "Please use Chrome or Edge browser for speech analysis",
        variant: "destructive",
      });
      return;
    }
    
    setIsRecording(true);
    setRecordingTime(0);
    setIsAnalyzing(true);
    setTranscript("");
    setInterimTranscript("");
    speechAnalyzerRef.current.reset();
    
    // Start speech recognition
    if (speechRecognitionRef.current) {
      const started = speechRecognitionRef.current.start();
      if (!started) {
        toast({
          title: "Speech Recognition Failed",
          description: "Could not start speech recognition",
          variant: "destructive",
        });
        return;
      }
    }
    
    toast({
      title: "Recording Started",
      description: "Real-time AI analysis and speech recognition active",
    });
  };

  const stopRecording = () => {
    setIsRecording(false);
    setIsAnalyzing(false);
    
    // Stop speech recognition
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.stop();
    }

    const finalAnalysis = speechAnalyzerRef.current.analyzeTranscript(transcript);
    
    toast({
      title: "Recording Stopped",
      description: "Analyzing your complete session...",
    });
    
    // Navigate to results with complete data
    setTimeout(() => {
      navigate("/results", { 
        state: { 
          duration: recordingTime,
          metrics: liveMetrics,
          transcript: transcript,
          speechAnalysis: finalAnalysis
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
                <canvas ref={canvasRef} className="absolute inset-0" />
                
                {!isCameraOn && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background/90">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-muted-foreground">Camera is off</p>
                    </div>
                  </div>
                )}

                {isAnalyzing && (
                  <div className="absolute top-4 left-4 flex flex-col gap-2">
                    <div className="flex items-center gap-2 px-3 py-2 bg-primary/90 rounded-full">
                      <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                      <span className="text-sm font-medium text-white">Live AI Analysis</span>
                    </div>
                    {(transcript || interimTranscript) && (
                      <div className="px-3 py-2 bg-background/90 rounded-lg max-w-md">
                        <p className="text-xs text-foreground">
                          {transcript} <span className="text-muted-foreground italic">{interimTranscript}</span>
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {isRecording && (
                  <div className="absolute top-4 right-4 px-4 py-2 bg-background/90 rounded-full">
                    <span className="text-lg font-bold text-primary">{formatTime(recordingTime)}</span>
                  </div>
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
                      >
                        <Video className="w-5 h-5 mr-2" />
                        Start Recording
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
              <h3 className="text-lg font-bold mb-4 text-foreground">Real-Time Analysis</h3>
              
              {!isAnalyzing ? (
                <div className="text-center py-8">
                  <Loader2 className="w-12 h-12 mx-auto mb-3 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Start recording to see live AI analysis
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Eye Contact */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Eye Contact</span>
                      <span className="text-sm font-bold text-primary">{liveMetrics.eyeContact}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-primary transition-all duration-500"
                        style={{ width: `${liveMetrics.eyeContact}%` }}
                      />
                    </div>
                  </div>

                  {/* Posture */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Posture</span>
                      <span className="text-sm font-bold text-primary">{liveMetrics.posture}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-secondary transition-all duration-500"
                        style={{ width: `${liveMetrics.posture}%` }}
                      />
                    </div>
                  </div>

                  {/* Voice Clarity */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Voice Clarity</span>
                      <span className="text-sm font-bold text-primary">{liveMetrics.clarity}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-primary transition-all duration-500"
                        style={{ width: `${liveMetrics.clarity}%` }}
                      />
                    </div>
                  </div>

                  {/* Engagement */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Engagement</span>
                      <span className="text-sm font-bold text-primary">{liveMetrics.engagement}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-secondary transition-all duration-500"
                        style={{ width: `${liveMetrics.engagement}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </Card>

            <Card className="p-6 bg-gradient-card border-border">
              <h3 className="text-lg font-bold mb-3 text-foreground">Quick Tips</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex gap-2">
                  <span className="text-accent">•</span>
                  Maintain eye contact with the camera
                </li>
                <li className="flex gap-2">
                  <span className="text-accent">•</span>
                  Keep your shoulders relaxed and straight
                </li>
                <li className="flex gap-2">
                  <span className="text-accent">•</span>
                  Speak clearly at a moderate pace
                </li>
                <li className="flex gap-2">
                  <span className="text-accent">•</span>
                  Use natural hand gestures
                </li>
              </ul>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Practice;
