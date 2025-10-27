import { Button } from "@/components/ui/button";
import { Presentation } from "lucide-react";

const Navigation = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center">
              <Presentation className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-foreground">AI Presentation Coach</span>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Features
            </a>
            <a href="#how-it-works" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              How It Works
            </a>
            <a href="#technology" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Technology
            </a>
            <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">
              Get Started
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
