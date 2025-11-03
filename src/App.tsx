import { Button } from '@/components/ui/button';
import { Mic, Eye, BarChart2, Lock } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

function App() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Hero Section (Demo Video Placeholder) */}
      <section className="relative h-screen overflow-hidden text-center text-white flex items-center justify-center">
        <video autoPlay muted loop playsInline className="absolute top-0 left-0 w-full h-full object-cover">
          <source src="YOUR_DEMO_VIDEO_URL.mp4" type="video/mp4" />
        </video>
        <div className="relative z-10 max-w-4xl mx-auto px-4">
          <h1 className="text-5xl font-bold mb-4">Master Public Speaking with Advanced AI/ML</h1>
          <p className="text-xl mb-8">Real-time AI analysis: voice, eye contact, posture. Powered by Mediapipe, Transformers.js, TF.js.</p>
          <Button size="lg" onClick={() => navigate('/coach')}>Start AI Coaching Session</Button>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 grid grid-cols-1 md:grid-cols-4 gap-8 container mx-auto">
        <div className="feature bg-card p-6 rounded-lg shadow-lg text-center">
          <Mic className="mx-auto mb-4 h-10 w-10 text-primary" />
          <strong>Voice Analysis</strong>
          <p>Advanced filler detection, pace, tone via NLP.</p>
        </div>
        <div className="feature bg-card p-6 rounded-lg shadow-lg text-center">
          <Eye className="mx-auto mb-4 h-10 w-10 text-primary" />
          <strong>Eye Contact Tracker</strong>
          <p>Mediapipe landmark-based gaze estimation.</p>
        </div>
        <div className="feature bg-card p-6 rounded-lg shadow-lg text-center">
          <BarChart2 className="mx-auto mb-4 h-10 w-10 text-primary" />
          <strong>Scorecard Feedback</strong>
          <p>ML-aggregated scores with TED-like tips.</p>
        </div>
        <div className="feature bg-card p-6 rounded-lg shadow-lg text-center">
          <Lock className="mx-auto mb-4 h-10 w-10 text-primary" />
          <strong>Private Practice</strong>
          <p>100% in-browser, no data shared.</p>
        </div>
      </section>

      {/* Footer (Removed Lovable) */}
      <footer className="py-6 bg-card text-center">
        <p>Built with ❤️ using Open-Source AI/ML Tools</p>
      </footer>
    </div>
  );
}

export default App;
