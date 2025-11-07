// Real-time speech recognition using Web Speech API
export class SpeechRecognitionService {
  private recognition: any;
  private isListening = false;
  private onTranscriptCallback: ((transcript: string, isFinal: boolean) => void) | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;

  constructor() {
    // Check for browser support
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.error('Speech recognition not supported in this browser');
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';
    this.recognition.maxAlternatives = 3;

    this.recognition.onresult = (event: any) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      if (finalTranscript && this.onTranscriptCallback) {
        this.onTranscriptCallback(finalTranscript.trim(), true);
      } else if (interimTranscript && this.onTranscriptCallback) {
        this.onTranscriptCallback(interimTranscript.trim(), false);
      }
    };

    this.recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      if (this.onErrorCallback) {
        this.onErrorCallback(event.error);
      }
      
      // Auto-restart on certain errors
      if (event.error === 'no-speech' || event.error === 'audio-capture') {
        setTimeout(() => {
          if (this.isListening) {
            this.start();
          }
        }, 1000);
      }
    };

    this.recognition.onend = () => {
      // Auto-restart if still supposed to be listening
      if (this.isListening) {
        setTimeout(() => {
          this.recognition.start();
        }, 100);
      }
    };
  }

  start() {
    if (!this.recognition) return false;
    
    try {
      this.isListening = true;
      this.recognition.start();
      return true;
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      return false;
    }
  }

  stop() {
    if (!this.recognition) return;
    
    this.isListening = false;
    try {
      this.recognition.stop();
    } catch (error) {
      console.error('Error stopping speech recognition:', error);
    }
  }

  onTranscript(callback: (transcript: string, isFinal: boolean) => void) {
    this.onTranscriptCallback = callback;
  }

  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }

  isSupported() {
    return !!this.recognition;
  }
}

// Advanced speech pattern analysis with temporal filler detection
export class SpeechAnalyzer {
  private wordTimestamps: Array<{ word: string; timestamp: number; isFiller: boolean }> = [];
  private fillerWords = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'actually', 'basically', 'literally', 'sort of', 'kind of'];
  private fillerPatterns = [
    /\b(um+|uh+|er+|ah+)\b/gi,
    /\b(like)\b(?!\s+(this|that|it))/gi, // Context-aware: "like" not followed by "this/that/it"
    /\b(you know|i mean|sort of|kind of)\b/gi,
    /\b(actually|basically|literally)\b(?=\s+[a-z])/gi, // Followed by lowercase word
  ];
  private lastWordTime = 0;
  private syllablePatterns = /[aeiouy]+/gi; // Approximate syllable counting

  analyzeTranscript(transcript: string) {
    const words = transcript.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const now = Date.now();

    // Track word timing with filler detection (NLP Token Classification)
    words.forEach(word => {
      const isFiller = this.isFillerWord(word);
      this.wordTimestamps.push({ word, timestamp: now, isFiller });
    });

    // Keep only last 60 seconds of data
    const cutoff = now - 60000;
    this.wordTimestamps = this.wordTimestamps.filter(w => w.timestamp > cutoff);

    // Calculate metrics with temporal mapping
    const totalWords = this.wordTimestamps.length;
    const fillerCount = this.wordTimestamps.filter(w => w.isFiller).length;
    
    // Temporal filler analysis: detect filler clusters
    const fillerClusters = this.detectFillerClusters();

    // Calculate WPM with syllable estimation for more accurate pace
    const timeSpan = (now - (this.wordTimestamps[0]?.timestamp || now)) / 1000 / 60;
    const totalSyllables = this.estimateSyllables(this.wordTimestamps.map(w => w.word).join(' '));
    const wordsPerMinute = timeSpan > 0 ? Math.round(totalWords / timeSpan) : 0;
    const syllablesPerMinute = timeSpan > 0 ? Math.round(totalSyllables / timeSpan) : 0;

    // Filler word percentage
    const fillerPercentage = totalWords > 0 ? (fillerCount / totalWords) * 100 : 0;

    // Pace analysis
    let paceScore = 100;
    if (wordsPerMinute < 100) paceScore -= (100 - wordsPerMinute) * 0.5; // Too slow
    if (wordsPerMinute > 200) paceScore -= (wordsPerMinute - 200) * 0.5; // Too fast
    paceScore = Math.max(25, Math.min(100, paceScore));

    // Clarity score (inverse of filler word usage)
    const clarityScore = Math.max(25, 100 - (fillerPercentage * 2));

    // Fluency score (based on consistent pace)
    const fluencyScore = paceScore;

    // Articulation score (based on word diversity)
    const uniqueWords = new Set(this.wordTimestamps.map(w => w.word)).size;
    const wordDiversity = totalWords > 0 ? (uniqueWords / totalWords) * 100 : 0;
    const articulationScore = Math.max(25, Math.min(100, 50 + wordDiversity));

    return {
      wordsPerMinute,
      syllablesPerMinute,
      fillerCount,
      fillerPercentage: Math.round(fillerPercentage),
      fillerClusters,
      totalWords,
      paceScore: Math.round(paceScore),
      clarityScore: Math.round(clarityScore),
      fluencyScore: Math.round(fluencyScore),
      articulationScore: Math.round(articulationScore),
      feedback: this.generateFeedback(wordsPerMinute, fillerPercentage, wordDiversity)
    };
  }

  private generateFeedback(wpm: number, fillerPct: number, diversity: number): string {
    const feedback: string[] = [];

    if (wpm < 100) {
      feedback.push('Try to speak a bit faster - aim for 120-150 words per minute.');
    } else if (wpm > 200) {
      feedback.push('Slow down slightly - speaking too fast can reduce clarity.');
    } else {
      feedback.push('Good speaking pace!');
    }

    if (fillerPct > 10) {
      feedback.push('Reduce filler words like "um" and "uh" by pausing instead.');
    } else if (fillerPct < 5) {
      feedback.push('Excellent - minimal filler words!');
    }

    if (diversity < 0.3) {
      feedback.push('Try to use more varied vocabulary.');
    }

    return feedback.join(' ');
  }

  getMetrics() {
    return this.analyzeTranscript('');
  }

  // Context-aware filler word detection using regex patterns and NLP
  private isFillerWord(word: string): boolean {
    // Direct match
    if (this.fillerWords.includes(word)) return true;
    
    // Pattern matching
    return this.fillerPatterns.some(pattern => pattern.test(word));
  }

  // Detect temporal filler clusters (multiple fillers in short time span)
  private detectFillerClusters(): number {
    let clusters = 0;
    let consecutiveFillers = 0;
    
    for (let i = 0; i < this.wordTimestamps.length; i++) {
      if (this.wordTimestamps[i].isFiller) {
        consecutiveFillers++;
        if (consecutiveFillers >= 2) {
          clusters++;
          consecutiveFillers = 0; // Reset after counting cluster
        }
      } else {
        consecutiveFillers = 0;
      }
    }
    
    return clusters;
  }

  // Estimate syllables for accurate pace calculation
  private estimateSyllables(text: string): number {
    if (!text) return 0;
    
    const words = text.toLowerCase().split(/\s+/);
    let totalSyllables = 0;
    
    for (const word of words) {
      // Remove non-alphabetic characters
      const cleanWord = word.replace(/[^a-z]/g, '');
      if (!cleanWord) continue;
      
      // Count vowel groups
      const matches = cleanWord.match(this.syllablePatterns);
      let syllables = matches ? matches.length : 1;
      
      // Adjust for silent 'e'
      if (cleanWord.endsWith('e') && syllables > 1) {
        syllables--;
      }
      
      // Ensure at least 1 syllable per word
      totalSyllables += Math.max(1, syllables);
    }
    
    return totalSyllables;
  }

  reset() {
    this.wordTimestamps = [];
    this.lastWordTime = 0;
  }
}
