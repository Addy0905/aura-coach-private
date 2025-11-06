/**
 * Real-time Speech Recognition & Analysis System
 * 
 * ALGORITHMS USED:
 * 1. Web Speech API - Browser-native speech-to-text
 * 2. Syllable Estimation - Phonetic pattern matching for accurate pace
 * 3. Temporal Clustering - Filler word grouping detection
 * 4. Context-Aware NLP - Smart filler word classification
 * 5. Lexical Diversity - Type-Token Ratio (TTR) for vocabulary analysis
 */

/**
 * Real-time Speech Recognition Service
 * Uses Web Speech API for browser-native speech-to-text conversion
 */
export class SpeechRecognitionService {
  private recognition: any = null;
  private isListening = false;
  private onTranscriptCallback: ((transcript: string, isFinal: boolean) => void) | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;
  private restartTimeout: NodeJS.Timeout | null = null;
  private startAttempts = 0;
  private readonly MAX_START_ATTEMPTS = 3;

  constructor() {
    this.initialize();
  }

  /**
   * Initialize Web Speech API
   * Checks browser compatibility and sets up recognition
   */
  private initialize(): void {
    try {
      // Check for browser support (Chrome, Edge, Safari)
      const SpeechRecognition = 
        (window as any).SpeechRecognition || 
        (window as any).webkitSpeechRecognition;
      
      if (!SpeechRecognition) {
        console.warn('Speech Recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
        return;
      }

      this.recognition = new SpeechRecognition();
      
      // Configuration for optimal speech recognition
      this.recognition.continuous = true; // Keep listening continuously
      this.recognition.interimResults = true; // Get partial results while speaking
      this.recognition.lang = 'en-US'; // Language
      this.recognition.maxAlternatives = 1; // Only need best result (optimized)
      
      this.setupEventHandlers();
      
    } catch (error) {
      console.error('Speech recognition initialization failed:', error);
    }
  }

  /**
   * Setup event handlers for speech recognition
   */
  private setupEventHandlers(): void {
    if (!this.recognition) return;

    // Handle speech results
    this.recognition.onresult = (event: any) => {
      try {
        let interimTranscript = '';
        let finalTranscript = '';

        // Process all results from the event
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const transcript = result[0].transcript;
          
          if (result.isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimTranscript += transcript;
          }
        }

        // Callback with final transcript (high priority)
        if (finalTranscript.trim() && this.onTranscriptCallback) {
          this.onTranscriptCallback(finalTranscript.trim(), true);
        } 
        // Callback with interim transcript (real-time feedback)
        else if (interimTranscript.trim() && this.onTranscriptCallback) {
          this.onTranscriptCallback(interimTranscript.trim(), false);
        }
      } catch (error) {
        console.error('Error processing speech results:', error);
      }
    };

    // Handle errors with smart recovery
    this.recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      
      if (this.onErrorCallback) {
        this.onErrorCallback(this.getErrorMessage(event.error));
      }
      
      // Smart auto-recovery for recoverable errors
      this.handleRecoverableError(event.error);
    };

    // Handle recognition end
    this.recognition.onend = () => {
      // Auto-restart if still supposed to be listening
      if (this.isListening && this.startAttempts < this.MAX_START_ATTEMPTS) {
        this.scheduleRestart();
      } else if (this.startAttempts >= this.MAX_START_ATTEMPTS) {
        console.warn('Max restart attempts reached. Please start recognition manually.');
        this.isListening = false;
        this.startAttempts = 0;
      }
    };

    // Handle recognition start
    this.recognition.onstart = () => {
      this.startAttempts = 0; // Reset on successful start
    };
  }

  /**
   * Get user-friendly error messages
   */
  private getErrorMessage(errorCode: string): string {
    const errorMessages: Record<string, string> = {
      'no-speech': 'No speech detected. Please speak into your microphone.',
      'audio-capture': 'Microphone not accessible. Please check permissions.',
      'not-allowed': 'Microphone permission denied. Please allow microphone access.',
      'network': 'Network error. Please check your internet connection.',
      'aborted': 'Speech recognition aborted.',
      'service-not-allowed': 'Speech recognition service not allowed.',
    };
    
    return errorMessages[errorCode] || `Speech recognition error: ${errorCode}`;
  }

  /**
   * Handle recoverable errors with exponential backoff
   */
  private handleRecoverableError(errorCode: string): void {
    const recoverableErrors = ['no-speech', 'audio-capture', 'network', 'aborted'];
    
    if (recoverableErrors.includes(errorCode) && this.isListening) {
      const delay = Math.min(1000 * Math.pow(2, this.startAttempts), 5000);
      this.scheduleRestart(delay);
    }
  }

  /**
   * Schedule recognition restart with delay
   */
  private scheduleRestart(delay: number = 100): void {
    if (this.restartTimeout) {
      clearTimeout(this.restartTimeout);
    }
    
    this.restartTimeout = setTimeout(() => {
      if (this.isListening) {
        this.startAttempts++;
        this.start();
      }
    }, delay);
  }

  /**
   * Start speech recognition
   */
  start(): boolean {
    if (!this.recognition) {
      console.error('Speech recognition not initialized');
      return false;
    }
    
    if (this.isListening) {
      console.warn('Speech recognition already running');
      return true;
    }
    
    try {
      this.isListening = true;
      this.recognition.start();
      return true;
    } catch (error: any) {
      // Handle "already started" error gracefully
      if (error.message?.includes('already started')) {
        return true;
      }
      console.error('Error starting speech recognition:', error);
      this.isListening = false;
      return false;
    }
  }

  /**
   * Stop speech recognition
   */
  stop(): void {
    if (!this.recognition) return;
    
    this.isListening = false;
    
    if (this.restartTimeout) {
      clearTimeout(this.restartTimeout);
      this.restartTimeout = null;
    }
    
    try {
      this.recognition.stop();
    } catch (error) {
      console.error('Error stopping speech recognition:', error);
    }
  }

  /**
   * Register callback for transcript updates
   */
  onTranscript(callback: (transcript: string, isFinal: boolean) => void): void {
    this.onTranscriptCallback = callback;
  }

  /**
   * Register callback for errors
   */
  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Check if speech recognition is supported
   */
  isSupported(): boolean {
    return this.recognition !== null;
  }

  /**
   * Get current listening state
   */
  getIsListening(): boolean {
    return this.isListening;
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.stop();
    this.onTranscriptCallback = null;
    this.onErrorCallback = null;
  }
}

/**
 * Advanced Speech Pattern Analysis
 * Analyzes speaking patterns, filler words, and delivery metrics
 */
export class SpeechAnalyzer {
  private wordTimestamps: Array<{ word: string; timestamp: number; isFiller: boolean }> = [];
  private readonly HISTORY_WINDOW_MS = 60000; // Keep 60 seconds of history
  
  /**
   * Comprehensive filler word lexicon
   * Categorized by type for better detection
   */
  private readonly fillerWords = new Set([
    // Hesitation fillers
    'um', 'uh', 'er', 'ah', 'eh', 'hmm', 'umm', 'uhh',
    // Discourse markers (context-dependent)
    'like', 'so', 'well', 'actually', 'basically', 'literally',
    'essentially', 'practically', 'virtually',
    // Phrases
    'you know', 'i mean', 'sort of', 'kind of', 'i think',
    'i guess', 'i suppose', 'you see', 'you know what i mean',
  ]);

  /**
   * Context-aware filler detection patterns
   * Uses regex for sophisticated matching
   */
  private readonly fillerPatterns: Array<RegExp> = [
    // Hesitation sounds (repeated vowels)
    /\b(um+|uh+|er+|ah+|eh+|hmm+)\b/gi,
    
    // "like" not used as a verb or comparison
    /\b(like)\b(?!\s+(this|that|it|to|you|me|him|her|us|them))/gi,
    
    // Discourse markers at start or with trailing words
    /\b(so|well)\b(?=\s+[a-z])/gi,
    
    // "Actually/basically/literally" used as fillers
    /\b(actually|basically|literally|essentially)\b(?=\s+[a-z])/gi,
    
    // Multi-word fillers
    /\b(you know|i mean|sort of|kind of|i think|i guess|you see)\b/gi,
  ];

  /**
   * Syllable pattern for phonetic analysis
   * Vowel clusters indicate syllables
   */
  private readonly syllablePattern = /[aeiouy]+/gi;

  /**
   * ALGORITHM 1: Comprehensive Transcript Analysis
   * Performs temporal analysis with NLP-based filler detection
   */
  analyzeTranscript(transcript: string): SpeechMetrics {
    if (!transcript || transcript.trim().length === 0) {
      return this.getDefaultMetrics();
    }

    try {
      const words = this.tokenizeWords(transcript);
      const now = Date.now();

      // Track word timing with filler detection
      words.forEach(word => {
        const isFiller = this.isFillerWord(word, transcript);
        this.wordTimestamps.push({ word, timestamp: now, isFiller });
      });

      // Maintain sliding window (last 60 seconds only)
      this.pruneHistory(now);

      if (this.wordTimestamps.length === 0) {
        return this.getDefaultMetrics();
      }

      // Calculate all metrics
      return this.calculateMetrics();
      
    } catch (error) {
      console.error('Speech analysis error:', error);
      return this.getDefaultMetrics();
    }
  }

  /**
   * ALGORITHM 2: Syllable Estimation (Phonetic Analysis)
   * Estimates syllables for accurate speaking rate calculation
   * 
   * How it works:
   * 1. Count vowel clusters (ae, io, uy groups)
   * 2. Adjust for silent 'e' at word end
   * 3. Ensure minimum 1 syllable per word
   * 
   * Example: "beautiful" → "eau", "i", "u" → 3 syllables
   */
  private estimateSyllables(text: string): number {
    if (!text) return 0;
    
    const words = text.toLowerCase().split(/\s+/);
    let totalSyllables = 0;
    
    for (const word of words) {
      // Clean word (remove non-letters)
      const cleanWord = word.replace(/[^a-z]/g, '');
      if (!cleanWord) continue;
      
      // Count vowel groups
      const matches = cleanWord.match(this.syllablePattern);
      let syllables = matches ? matches.length : 1;
      
      // Adjust for silent 'e' (e.g., "make" has 1 syllable, not 2)
      if (cleanWord.endsWith('e') && syllables > 1 && cleanWord.length > 2) {
        syllables--;
      }
      
      // Adjust for 'le' ending (e.g., "table" → "ta-ble" = 2)
      if (cleanWord.endsWith('le') && syllables > 1 && cleanWord.length > 2) {
        syllables++;
      }
      
      // Minimum 1 syllable per word
      totalSyllables += Math.max(1, syllables);
    }
    
    return totalSyllables;
  }

  /**
   * ALGORITHM 3: Context-Aware Filler Detection
   * Uses NLP patterns to distinguish filler usage from legitimate usage
   * 
   * Example:
   * - "I like pizza" → "like" is a verb (NOT filler)
   * - "It's like really good" → "like" is a filler
   */
  private isFillerWord(word: string, fullContext: string): boolean {
    const lowerWord = word.toLowerCase();
    
    // Direct lexicon match
    if (this.fillerWords.has(lowerWord)) {
      // Context check for ambiguous words
      if (lowerWord === 'like') {
        return this.isLikeFiller(word, fullContext);
      }
      return true;
    }
    
    // Pattern matching for complex cases
    return this.fillerPatterns.some(pattern => {
      pattern.lastIndex = 0; // Reset regex
      return pattern.test(lowerWord);
    });
  }

  /**
   * Context analysis for "like" as filler vs verb
   */
  private isLikeFiller(word: string, context: string): boolean {
    const wordIndex = context.toLowerCase().indexOf(word.toLowerCase());
    if (wordIndex === -1) return false;
    
    // Get next few words for context
    const afterWord = context.substring(wordIndex + word.length, wordIndex + word.length + 30).toLowerCase();
    
    // "like" followed by object pronouns or "to" is likely a verb
    const verbIndicators = ['this', 'that', 'it', 'you', 'me', 'him', 'her', 'to'];
    const hasVerbIndicator = verbIndicators.some(indicator => 
      afterWord.trim().startsWith(indicator)
    );
    
    return !hasVerbIndicator;
  }

  /**
   * ALGORITHM 4: Temporal Filler Clustering
   * Detects consecutive filler words (indicates hesitation patterns)
   * 
   * Example: "um... well... uh... I think" → 3-filler cluster
   */
  private detectFillerClusters(): number {
    let clusters = 0;
    let consecutiveFillers = 0;
    
    for (const entry of this.wordTimestamps) {
      if (entry.isFiller) {
        consecutiveFillers++;
      } else {
        // Count cluster if 2+ consecutive fillers
        if (consecutiveFillers >= 2) {
          clusters++;
        }
        consecutiveFillers = 0;
      }
    }
    
    // Check last sequence
    if (consecutiveFillers >= 2) {
      clusters++;
    }
    
    return clusters;
  }

  /**
   * ALGORITHM 5: Lexical Diversity (Type-Token Ratio)
   * Measures vocabulary richness
   * TTR = (unique words / total words) × 100
   * 
   * Higher TTR = more diverse vocabulary
   */
  private calculateLexicalDiversity(): number {
    if (this.wordTimestamps.length === 0) return 0;
    
    const nonFillerWords = this.wordTimestamps
      .filter(w => !w.isFiller)
      .map(w => w.word.toLowerCase());
    
    if (nonFillerWords.length === 0) return 0;
    
    const uniqueWords = new Set(nonFillerWords).size;
    return (uniqueWords / nonFillerWords.length) * 100;
  }

  /**
   * Calculate comprehensive speech metrics
   */
  private calculateMetrics(): SpeechMetrics {
    const totalWords = this.wordTimestamps.length;
    const fillerCount = this.wordTimestamps.filter(w => w.isFiller).length;
    const contentWords = totalWords - fillerCount;
    
    // Time calculation
    const firstTimestamp = this.wordTimestamps[0]?.timestamp || Date.now();
    const lastTimestamp = this.wordTimestamps[this.wordTimestamps.length - 1]?.timestamp || Date.now();
    const timeSpanMinutes = (lastTimestamp - firstTimestamp) / 1000 / 60;
    const timeSpanSeconds = (lastTimestamp - firstTimestamp) / 1000;
    
    // Words per minute
    const wordsPerMinute = timeSpanMinutes > 0 ? Math.round(contentWords / timeSpanMinutes) : 0;
    
    // Syllables per minute (more accurate than WPM for pace)
    const allText = this.wordTimestamps.map(w => w.word).join(' ');
    const totalSyllables = this.estimateSyllables(allText);
    const syllablesPerMinute = timeSpanMinutes > 0 ? Math.round(totalSyllables / timeSpanMinutes) : 0;
    
    // Filler analysis
    const fillerPercentage = totalWords > 0 ? (fillerCount / totalWords) * 100 : 0;
    const fillerClusters = this.detectFillerClusters();
    
    // Pace score (optimal: 120-160 WPM)
    const paceScore = this.calculatePaceScore(wordsPerMinute);
    
    // Clarity score (based on filler usage)
    const clarityScore = this.calculateClarityScore(fillerPercentage);
    
    // Fluency score (consistency of pace)
    const fluencyScore = this.calculateFluencyScore(paceScore, fillerPercentage);
    
    // Articulation score (vocabulary diversity)
    const lexicalDiversity = this.calculateLexicalDiversity();
    const articulationScore = this.calculateArticulationScore(lexicalDiversity);
    
    // Generate feedback
    const feedback = this.generateFeedback(wordsPerMinute, fillerPercentage, lexicalDiversity);
    
    return {
      wordsPerMinute,
      syllablesPerMinute,
      fillerCount,
      fillerPercentage: Math.round(fillerPercentage * 10) / 10,
      fillerClusters,
      totalWords,
      contentWords,
      paceScore: Math.round(paceScore),
      clarityScore: Math.round(clarityScore),
      fluencyScore: Math.round(fluencyScore),
      articulationScore: Math.round(articulationScore),
      lexicalDiversity: Math.round(lexicalDiversity),
      timeSpoken: Math.round(timeSpanSeconds),
      feedback,
    };
  }

  /**
   * Calculate pace score (optimal range: 120-160 WPM)
   */
  private calculatePaceScore(wpm: number): number {
    if (wpm === 0) return 0;
    
    const optimalMin = 120;
    const optimalMax = 160;
    
    if (wpm >= optimalMin && wpm <= optimalMax) {
      return 100;
    } else if (wpm < optimalMin) {
      // Too slow: score decreases as WPM decreases below 120
      return Math.max(0, 100 - ((optimalMin - wpm) * 0.8));
    } else {
      // Too fast: score decreases as WPM increases above 160
      return Math.max(0, 100 - ((wpm - optimalMax) * 0.6));
    }
  }

  /**
   * Calculate clarity score (lower filler % = higher clarity)
   */
  private calculateClarityScore(fillerPct: number): number {
    // Excellent: < 5%
    // Good: 5-10%
    // Fair: 10-15%
    // Poor: > 15%
    return Math.max(0, Math.min(100, 100 - (fillerPct * 5)));
  }

  /**
   * Calculate fluency score (combined pace + clarity)
   */
  private calculateFluencyScore(paceScore: number, fillerPct: number): number {
    const clarityFactor = Math.max(0, 1 - (fillerPct / 20));
    return Math.round(paceScore * clarityFactor);
  }

  /**
   * Calculate articulation score (vocabulary diversity)
   */
  private calculateArticulationScore(lexicalDiversity: number): number {
    // Excellent: > 70%
    // Good: 50-70%
    // Fair: 30-50%
    // Poor: < 30%
    return Math.max(25, Math.min(100, lexicalDiversity + 30));
  }

  /**
   * Generate actionable feedback
   */
  private generateFeedback(wpm: number, fillerPct: number, diversity: number): string {
    const feedback: string[] = [];

    // Pace feedback
    if (wpm === 0) {
      feedback.push('Start speaking to analyze your pace.');
    } else if (wpm < 100) {
      feedback.push('🐢 Speak faster - aim for 120-160 words per minute.');
    } else if (wpm > 180) {
      feedback.push('🏃 Slow down - speaking too fast reduces clarity.');
    } else if (wpm >= 120 && wpm <= 160) {
      feedback.push('✓ Perfect pace!');
    } else {
      feedback.push('✓ Good pace!');
    }

    // Filler feedback
    if (fillerPct > 15) {
      feedback.push('⚠️ Too many filler words - pause instead of saying "um" or "uh".');
    } else if (fillerPct > 10) {
      feedback.push('Reduce filler words by taking brief pauses.');
    } else if (fillerPct < 5) {
      feedback.push('✓ Excellent - minimal filler words!');
    }

    // Diversity feedback
    if (diversity < 30) {
      feedback.push('Try using more varied vocabulary.');
    } else if (diversity > 60) {
      feedback.push('✓ Great vocabulary diversity!');
    }

    return feedback.join(' ');
  }

  /**
   * Tokenize text into words
   */
  private tokenizeWords(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s'-]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }

  /**
   * Prune old entries from history (keep last 60 seconds)
   */
  private pruneHistory(currentTime: number): void {
    const cutoff = currentTime - this.HISTORY_WINDOW_MS;
    this.wordTimestamps = this.wordTimestamps.filter(w => w.timestamp > cutoff);
  }

  /**
   * Get current metrics without new transcript
   */
  getMetrics(): SpeechMetrics {
    return this.wordTimestamps.length > 0 
      ? this.calculateMetrics() 
      : this.getDefaultMetrics();
  }

  /**
   * Get default metrics (zeros)
   */
  private getDefaultMetrics(): SpeechMetrics {
    return {
      wordsPerMinute: 0,
      syllablesPerMinute: 0,
      fillerCount: 0,
      fillerPercentage: 0,
      fillerClusters: 0,
      totalWords: 0,
      contentWords: 0,
      paceScore: 0,
      clarityScore: 0,
      fluencyScore: 0,
      articulationScore: 0,
      lexicalDiversity: 0,
      timeSpoken: 0,
      feedback: 'Start speaking to begin analysis.',
    };
  }

  /**
   * Reset analyzer state
   */
  reset(): void {
    this.wordTimestamps = [];
  }

  /**
   * Get detailed filler word breakdown
   */
  getFillerBreakdown(): { [word: string]: number } {
    const breakdown: { [word: string]: number } = {};
    
    this.wordTimestamps
      .filter(w => w.isFiller)
      .forEach(w => {
        breakdown[w.word] = (breakdown[w.word] || 0) + 1;
      });
    
    return breakdown;
  }
}

/**
 * Speech metrics interface
 */
export interface SpeechMetrics {
  wordsPerMinute: number;
  syllablesPerMinute: number;
  fillerCount: number;
  fillerPercentage: number;
  fillerClusters: number;
  totalWords: number;
  contentWords: number;
  paceScore: number;
  clarityScore: number;
  fluencyScore: number;
  articulationScore: number;
  lexicalDiversity: number;
  timeSpoken: number;
  feedback: string;
}
