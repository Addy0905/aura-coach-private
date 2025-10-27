import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Download, Share2, TrendingUp, Award } from "lucide-react";
import { Progress } from "@/components/ui/progress";

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { duration = 0, metrics = {} } = location.state || {};

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const overallScore = Math.round(
    (metrics.eyeContact + metrics.posture + metrics.clarity + metrics.engagement) / 4
  );

  const getFeedback = (score: number) => {
    if (score >= 80) return { text: "Excellent!", color: "text-green-400" };
    if (score >= 60) return { text: "Good Work", color: "text-blue-400" };
    if (score >= 40) return { text: "Keep Practicing", color: "text-yellow-400" };
    return { text: "Needs Improvement", color: "text-orange-400" };
  };

  const recommendations = [
    {
      title: "Eye Contact",
      score: metrics.eyeContact,
      feedback: "Maintain consistent eye contact with the camera. Look directly at the lens when speaking.",
    },
    {
      title: "Posture",
      score: metrics.posture,
      feedback: "Keep your shoulders back and maintain an upright posture throughout your presentation.",
    },
    {
      title: "Voice Clarity",
      score: metrics.clarity,
      feedback: "Speak at a moderate pace and enunciate clearly. Avoid filler words like 'um' and 'uh'.",
    },
    {
      title: "Engagement",
      score: metrics.engagement,
      feedback: "Use varied vocal tones and natural gestures to maintain audience engagement.",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-hero p-4">
      <div className="container mx-auto max-w-6xl">
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/practice")}
            className="text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Practice
          </Button>
          
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="border-border">
              <Download className="w-4 h-4 mr-2" />
              Export PDF
            </Button>
            <Button variant="outline" size="sm" className="border-border">
              <Share2 className="w-4 h-4 mr-2" />
              Share
            </Button>
          </div>
        </div>

        {/* Overall Score Card */}
        <Card className="p-8 bg-gradient-card border-border mb-6 text-center">
          <div className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-gradient-primary mb-4 relative">
            <div className="absolute inset-2 rounded-full bg-background flex items-center justify-center">
              <span className="text-4xl font-bold text-primary">{overallScore}%</span>
            </div>
          </div>
          
          <h2 className="text-3xl font-bold mb-2 text-foreground">
            Session Complete!
          </h2>
          <p className={`text-xl mb-2 ${getFeedback(overallScore).color}`}>
            {getFeedback(overallScore).text}
          </p>
          <p className="text-muted-foreground">
            Duration: {formatTime(duration)}
          </p>
        </Card>

        {/* Detailed Metrics */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {recommendations.map((item, index) => (
            <Card
              key={index}
              className="p-6 bg-gradient-card border-border animate-fade-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-foreground">{item.title}</h3>
                <span className={`text-2xl font-bold ${getFeedback(item.score).color}`}>
                  {item.score}%
                </span>
              </div>
              
              <Progress value={item.score} className="mb-4" />
              
              <p className="text-sm text-muted-foreground leading-relaxed">
                {item.feedback}
              </p>
            </Card>
          ))}
        </div>

        {/* Key Insights */}
        <Card className="p-6 bg-gradient-card border-border mb-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-accent" />
            <h3 className="text-lg font-bold text-foreground">Key Insights</h3>
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-background/50">
              <div className="text-2xl font-bold text-primary mb-1">
                {Math.round(duration / 60)} min
              </div>
              <div className="text-sm text-muted-foreground">Practice Time</div>
            </div>
            
            <div className="p-4 rounded-lg bg-background/50">
              <div className="text-2xl font-bold text-accent mb-1">
                {Math.round((metrics.eyeContact / 100) * duration)}s
              </div>
              <div className="text-sm text-muted-foreground">Good Eye Contact</div>
            </div>
            
            <div className="p-4 rounded-lg bg-background/50">
              <div className="text-2xl font-bold text-primary mb-1">
                95%
              </div>
              <div className="text-sm text-muted-foreground">AI Confidence</div>
            </div>
          </div>
        </Card>

        {/* Achievement Badge */}
        {overallScore >= 70 && (
          <Card className="p-6 bg-gradient-primary border-primary/50 text-center animate-fade-in">
            <Award className="w-12 h-12 mx-auto mb-3 text-white" />
            <h3 className="text-xl font-bold text-white mb-2">
              Achievement Unlocked!
            </h3>
            <p className="text-white/90">
              You've earned the "Rising Speaker" badge for scoring above 70%
            </p>
          </Card>
        )}

        {/* Action Buttons */}
        <div className="flex justify-center gap-4 mt-8">
          <Button
            size="lg"
            onClick={() => navigate("/practice")}
            className="bg-primary hover:bg-primary/90"
          >
            Practice Again
          </Button>
          <Button
            size="lg"
            variant="outline"
            onClick={() => navigate("/")}
            className="border-border"
          >
            Back to Home
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Results;
