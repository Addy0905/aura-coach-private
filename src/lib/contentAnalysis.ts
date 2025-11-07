// Advanced content analysis: USE, TF-IDF, Cosine Similarity, Sentiment, NER
// Browser-compatible NLP using lightweight algorithms

export interface ContentMetrics {
  coherenceScore: number; // 0-100, semantic consistency
  keywordRelevance: number; // 0-100, topic relevance via TF-IDF
  sentimentScore: number; // -100 to 100, VADER-like sentiment
  sentimentLabel: string; // 'positive', 'neutral', 'negative'
  entityCount: number; // Number of named entities detected
  topKeywords: string[]; // Top 5 keywords by TF-IDF
  topEntities: string[]; // Top entities (names, places, orgs)
}

export class ContentAnalyzer {
  private documentHistory: string[] = [];
  private vocabularyIDF: Map<string, number> = new Map();
  
  // VADER-like sentiment lexicon (simplified)
  private sentimentLexicon = new Map<string, number>([
    // Positive words
    ['good', 3], ['great', 4], ['excellent', 5], ['amazing', 5], ['wonderful', 4],
    ['fantastic', 4], ['love', 3], ['perfect', 4], ['best', 4], ['beautiful', 3],
    ['happy', 3], ['excited', 3], ['confident', 4], ['success', 3], ['win', 3],
    // Negative words
    ['bad', -3], ['terrible', -4], ['awful', -4], ['horrible', -4], ['worst', -4],
    ['hate', -3], ['fail', -3], ['problem', -2], ['issue', -2], ['difficult', -2],
    ['sad', -3], ['angry', -3], ['frustrated', -2], ['confused', -2],
    // Modifiers
    ['very', 1.5], ['really', 1.5], ['extremely', 2], ['absolutely', 2],
    ['not', -1], ['never', -1.5], ['no', -1],
  ]);
  
  // Common stopwords for TF-IDF
  private stopwords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i',
    'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just', 'so'
  ]);

  analyzeContent(transcript: string, topic?: string): ContentMetrics {
    if (!transcript || transcript.length < 10) {
      return this.getDefaultMetrics();
    }

    // Add to document history for IDF calculation
    this.documentHistory.push(transcript);
    if (this.documentHistory.length > 20) {
      this.documentHistory.shift();
    }
    this.updateIDF();

    // Tokenize and clean
    const tokens = this.tokenize(transcript);
    
    // Calculate coherence using cosine similarity between sentences
    const coherence = this.calculateCoherence(transcript);
    
    // Calculate keyword relevance using TF-IDF
    const tfIdfScores = this.calculateTFIDF(tokens);
    const topKeywords = Array.from(tfIdfScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([word]) => word);
    
    const keywordRelevance = topic 
      ? this.calculateTopicRelevance(tokens, topic)
      : Math.min(100, tfIdfScores.size * 5);
    
    // Sentiment analysis (VADER-like)
    const sentiment = this.analyzeSentiment(tokens);
    
    // Named Entity Recognition (simple pattern-based)
    const entities = this.extractEntities(transcript);
    
    return {
      coherenceScore: Math.round(coherence),
      keywordRelevance: Math.round(keywordRelevance),
      sentimentScore: Math.round(sentiment.score),
      sentimentLabel: sentiment.label,
      entityCount: entities.length,
      topKeywords,
      topEntities: entities.slice(0, 5),
    };
  }

  // Tokenize text into words
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2 && !this.stopwords.has(word));
  }

  // Calculate semantic coherence using cosine similarity between sentences
  private calculateCoherence(text: string): number {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
    
    if (sentences.length < 2) return 75; // Single sentence = reasonable coherence
    
    let totalSimilarity = 0;
    let comparisons = 0;
    
    for (let i = 0; i < sentences.length - 1; i++) {
      const sim = this.cosineSimilarity(
        this.tokenize(sentences[i]),
        this.tokenize(sentences[i + 1])
      );
      totalSimilarity += sim;
      comparisons++;
    }
    
    const avgSimilarity = comparisons > 0 ? totalSimilarity / comparisons : 0;
    return Math.max(25, Math.min(100, avgSimilarity * 100));
  }

  // Cosine similarity between two token arrays
  private cosineSimilarity(tokens1: string[], tokens2: string[]): number {
    const set1 = new Set(tokens1);
    const set2 = new Set(tokens2);
    
    // Intersection
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    
    // Magnitude
    const magnitude1 = Math.sqrt(set1.size);
    const magnitude2 = Math.sqrt(set2.size);
    
    if (magnitude1 === 0 || magnitude2 === 0) return 0;
    
    return intersection.size / (magnitude1 * magnitude2);
  }

  // TF-IDF calculation
  private calculateTFIDF(tokens: string[]): Map<string, number> {
    const tfIdf = new Map<string, number>();
    const termFreq = new Map<string, number>();
    
    // Calculate TF (Term Frequency)
    tokens.forEach(token => {
      termFreq.set(token, (termFreq.get(token) || 0) + 1);
    });
    
    // Calculate TF-IDF
    termFreq.forEach((tf, term) => {
      const idf = this.vocabularyIDF.get(term) || 1;
      tfIdf.set(term, (tf / tokens.length) * idf);
    });
    
    return tfIdf;
  }

  // Update IDF (Inverse Document Frequency) from document history
  private updateIDF() {
    const docCount = this.documentHistory.length;
    if (docCount === 0) return;
    
    const termDocCount = new Map<string, number>();
    
    this.documentHistory.forEach(doc => {
      const uniqueTerms = new Set(this.tokenize(doc));
      uniqueTerms.forEach(term => {
        termDocCount.set(term, (termDocCount.get(term) || 0) + 1);
      });
    });
    
    termDocCount.forEach((count, term) => {
      this.vocabularyIDF.set(term, Math.log(docCount / count));
    });
  }

  // Calculate relevance to a specific topic
  private calculateTopicRelevance(tokens: string[], topic: string): number {
    const topicTokens = this.tokenize(topic);
    const matchCount = tokens.filter(token => topicTokens.includes(token)).length;
    return Math.min(100, (matchCount / Math.max(topicTokens.length, 1)) * 100);
  }

  // VADER-like sentiment analysis
  private analyzeSentiment(tokens: string[]): { score: number; label: string } {
    let score = 0;
    let modifier = 1;
    
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const sentimentValue = this.sentimentLexicon.get(token);
      
      if (sentimentValue !== undefined) {
        if (Math.abs(sentimentValue) < 2) {
          // It's a modifier
          modifier = sentimentValue;
        } else {
          // It's a sentiment word
          score += sentimentValue * modifier;
          modifier = 1; // Reset modifier
        }
      }
    }
    
    // Normalize score to -100 to 100
    const normalizedScore = Math.max(-100, Math.min(100, score * 5));
    
    let label = 'neutral';
    if (normalizedScore > 20) label = 'positive';
    else if (normalizedScore < -20) label = 'negative';
    
    return { score: normalizedScore, label };
  }

  // Simple Named Entity Recognition (pattern-based)
  private extractEntities(text: string): string[] {
    const entities: string[] = [];
    
    // Capitalized words (potential names, places, organizations)
    const capitalizedPattern = /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g;
    const matches = text.match(capitalizedPattern);
    
    if (matches) {
      // Deduplicate and filter common words
      const uniqueEntities = new Set(
        matches.filter(entity => 
          !['I', 'The', 'A', 'An', 'This', 'That'].includes(entity)
        )
      );
      entities.push(...uniqueEntities);
    }
    
    return entities;
  }

  private getDefaultMetrics(): ContentMetrics {
    return {
      coherenceScore: 25,
      keywordRelevance: 25,
      sentimentScore: 0,
      sentimentLabel: 'neutral',
      entityCount: 0,
      topKeywords: [],
      topEntities: [],
    };
  }

  reset() {
    this.documentHistory = [];
    this.vocabularyIDF.clear();
  }
}
