import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { FilesetResolver, FaceLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { pipeline } from '@huggingface/transformers';

const CoachSession = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [feedback, setFeedback] = useState({ fillers: 0, pace: 0, sentiment: '', eyeContact: false, posture: 'Good' });
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  let faceLandmarker: FaceLandmarker | null = null;
  let poseLandmarker: PoseLandmarker | null = null;
  let asrPipeline: any = null;
  let sentimentPipeline: any = null;

  useEffect(() => {
    const initAI = async () => {
      // Init Mediapipe
      const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm');
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
        runningMode: 'VIDEO', numFaces: 1,
      });
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/pose_landmarker.task' },
        runningMode: 'VIDEO',
      });

      // Init HuggingFace
      asrPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small', { dtype: 'q4' });
      sentimentPipeline = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');

      // Setup Webcam
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    };
    initAI();

    return () => { faceLandmarker?.close(); poseLandmarker?.close(); };
  }, []);

  const startSession = () => {
    setIsRecording(true);
    const stream = videoRef.current?.srcObject as MediaStream;
    mediaRecorderRef.current = new MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = (e) => chunksRef.current.push(e.data);
    mediaRecorderRef.current.start(1000); // Chunk every second for real-time feel
    requestAnimationFrame(processFrame);
  };

  const stopSession = async () => {
    setIsRecording(false);
    mediaRecorderRef.current?.stop();
    const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
    const arrayBuffer = await audioBlob.arrayBuffer();

    // ASR & Analysis
    const transcription = await asrPipeline(arrayBuffer);
    const text = transcription.text;
    const fillersCount = (text.match(/\b(um|uh|like|you know)\b/gi) || []).length;
    const pace = text.split(' ').length / (audioBlob.size / 1000000 * 60); // Approx WPM
    const sentimentResult = await sentimentPipeline(text);

    setFeedback((prev) => ({
      ...prev,
      fillers: fillersCount,
      pace,
      sentiment: `${sentimentResult[0].label} (${sentimentResult[0].score.toFixed(2)})`,
    }));
    chunksRef.current = [];
  };

  const processFrame = () => {
    if (!videoRef.current || !canvasRef.current || !isRecording) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const faceResult = faceLandmarker?.detectForVideo(video, performance.now());
    const poseResult = poseLandmarker?.detectForVideo(video, performance.now());

    // Eye Contact (Advanced: Check if eyes are centered)
    let eyeContact = false;
    if (faceResult?.faceLandmarks[0]) {
      const landmarks = faceResult.faceLandmarks[0];
      const leftEye = (landmarks[159].x + landmarks[145].x) / 2; // Avg left eye X
      const rightEye = (landmarks[386].x + landmarks[374].x) / 2;
      eyeContact = Math.abs(leftEye - 0.4) < 0.1 && Math.abs(rightEye - 0.6) < 0.1; // Centered gaze approx
    }

    // Posture (Advanced: Check shoulder alignment for slouch)
    let posture = 'Good';
    if (poseResult?.landmarks[0]) {
      const landmarks = poseResult.landmarks[0];
      const leftShoulderY = landmarks[11].y;
      const rightShoulderY = landmarks[12].y;
      if (Math.abs(leftShoulderY - rightShoulderY) > 0.05) posture = 'Slouching - Straighten up!';
    }

    // Draw on canvas (optional visuals)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // Add drawing code for landmarks if desired...

    setFeedback((prev) => ({ ...prev, eyeContact, posture }));

    requestAnimationFrame(processFrame);
  };

  return (
    <Card className="p-6">
      <video ref={videoRef} autoPlay muted playsInline className="w-full" />
      <canvas ref={canvasRef} className="w-full" />
      {!isRecording ? (
        <Button onClick={startSession}>Start Session</Button>
      ) : (
        <Button onClick={stopSession}>Stop & Analyze</Button>
      )}
      <div className="mt-4">
        <h2>Real-Time Feedback</h2>
        <p>Eye Contact: {feedback.eyeContact ? 'Yes' : 'No'}</p>
        <p>Posture: {feedback.posture}</p>
        <p>Fillers Detected: {feedback.fillers}</p>
        <p>Pace (WPM): {feedback.pace.toFixed(0)}</p>
        <p>Tone: {feedback.sentiment}</p>
      </div>
    </Card>
  );
};

export default CoachSession;
