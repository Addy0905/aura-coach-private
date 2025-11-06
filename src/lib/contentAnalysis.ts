/**
 * Advanced Content Analysis using NLP Algorithms (Optimized for Real-time & Accuracy)
 *
 * ALGORITHMS USED:
 * 1. TF-IDF (Term Frequency-Inverse Document Frequency) - Keyword extraction
 * 2. Cosine Similarity - Semantic coherence measurement
 * 3. VADER Sentiment Analysis - Emotion/opinion detection
 * 4. Named Entity Recognition (NER) - Entity extraction
 * 5. Jaccard Similarity - Text overlap measurement
 */
export interface ContentMetrics {
  coherenceScore: number; // 0-100, semantic consistency between sentences
  keywordRelevance: number; // 0-100, topic relevance via TF-IDF
  sentimentScore: number; // -100 to 100, VADER-like sentiment
  sentimentLabel: string; // 'positive', 'neutral', 'negative'
  entityCount: number; // Number of named entities detected
  topKeywords: string[]; // Top 5 keywords by TF-IDF score
  topEntities: string[]; // Top entities (names, places, organizations)
  vocabularyRichness: number;-ns 0-100, unique words / total words
  readabilityScore: number; // 0-100, based on sentence complexity
}

export class ContentAnalyzer {
  private documentHistory: string[] = [];
  private vocabularyIDF: Map<string, number> = new Map();
  private readonly MAX_HISTORY = 20;

  /**
   * VADER Sentiment Lexicon (Valence Aware Dictionary and sEntiment Reasoner)
   * Assigns sentiment scores to words: positive (1-5), negative (-1 to -5)
   */
  private readonly sentimentLexicon = new Map<string, number>([
    // Strong positive (5)
    ['amazing', 5], ['excellent', 5], ['outstanding', 5], ['perfect', 5],
    ['brilliant', 5], ['exceptional', 5], ['superb', 5], ['magnificent', 5],
   
    // Positive (3-4)
    ['good', 3], ['great', 4], ['wonderful', 4], ['fantastic', 4], ['awesome', 4],
    ['love', 3], ['best', 4], ['beautiful', 3], ['happy', 3], ['excited', 3],
    ['confident', 4], ['success', 3], ['win', 3], ['nice', 3], ['enjoy', 3],
    ['pleased', 3], ['delighted', 4], ['proud', 3], ['thrilled', 4],
   
    // Strong negative (-5 to -4)
    ['terrible', -4], ['awful', -4], ['horrible', -4], ['worst', -4],
    ['disgusting', -5], ['hate', -4], ['disaster', -5], ['nightmare', -5],
   
    // Negative (-3 to -2)
    ['bad', -3], ['fail', -3], ['problem', -2], ['issue', -2], ['difficult', -2],
    ['sad', -3], ['angry', -3], ['frustrated', -2], ['confused', -2], ['weak', -2],
    ['poor', -2], ['wrong', -2], ['mistake', -2], ['concerned', -2],
   
    // Modifiers (amplifiers and diminishers)
    ['very', 1.5], ['really', 1.5], ['extremely', 2], ['absolutely', 2],
    ['incredibly', 2], ['totally', 1.8], ['completely', 1.8],
    ['not', -1], ['never', -1.5], ['no', -1], ['hardly', -0.8], ['barely', -0.8],
   
    // Context modifiers
    ['but', 0.5], ['however', 0.5], ['although', 0.3],
  ]);

  /**
   * Stopwords - Common words to filter out from analysis
   */
  private readonly stopwords = new Set([
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'is', 'am', 'are', 'was', 'were', 'been', 'be', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'much', 'many'
  ]);

  /**
   * Main analysis function - processes transcript and returns comprehensive metrics
   */
  analyzeContent(transcript: string, topic?: string): ContentMetrics {
    if (!transcript || transcript.trim().length < 10) {
      return this.getDefaultMetrics();
    }

    const cleanTranscript = transcript.trim().toLowerCase();
    
    try {
      // Update document history for IDF
      this.updateDocumentHistory(cleanTranscript);

      // Tokenize once and reuse
      const tokens = this.tokenize(cleanTranscript);
      if (tokens.length === 0) return this.getDefaultMetrics();

      // Pre-split sentences for reuse
      const sentences = cleanTranscript
        .split(/[.!?]+/)
        .map(s => s.trim())
        .filter(s => s.length > 10);

      // 1. COHERENCE
      const coherence = this.calculateCoherence(sentences);

      // 2. TF-IDF + Keywords
      const tfIdfScores = this.calculateTFIDF(tokens);
      const topKeywords = this.extractTopKeywords(tfIdfScores, 5);

      // 3. KEYWORD RELEVANCE
      const keywordRelevance = topic
        ? this.calculateTopicRelevance(tokens, topic.toLowerCase())
        : this.calculateOverallRelevance(tfIdfScores, tokens.length);

      // 4. SENTIMENT
      const sentiment = this.analyzeSentiment(cleanTranscript);

      // 5. NER
      const entities = this.extractEntities(transcript); // Use original case

      // 6. VOCABULARY RICHNESS
      const vocabularyRichness = this.calculateVocabularyRichness(tokens);

      // 7. READABILITY
      const readabilityScore = this.calculateReadability(transcript);

      return {
        coherenceScore: Math.round(Math.max(0, Math.min(100, coherence))),
        keywordRelevance: Math.round(Math.max(0, Math.min(100, keywordRelevance))),
        sentimentScore: Math.round(Math.max(-100, Math.min(100, sentiment.score))),
        sentimentLabel: sentiment.label,
        entityCount: entities.length,
        topKeywords,
        topEntities: entities.slice(0, 5),
        vocabularyRichness: Math.round(Math.max(0, Math.min(100, vocabularyRichness))),
        readabilityScore: Math.round(Math.max(0, Math.min(100, readabilityScore))),
      };
    } catch (error) {
      console.error('Content analysis error:', error);
      return this.getDefaultMetrics();
    }
  }

  /**
   * Tokenization - optimized with single pass
   */
  private tokenize(text: string): string[] {
    const words: string[] = [];
    const seen = new Set<string>();
    let current = '';

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (/[a-z0-9'-]/.test(char)) {
        current += char;
      } else if (current) {
        if (current.length > 2 && !this.stopwords.has(current) && !/^\d+$/.test(current)) {
          if (!seen.has(current)) {
            words.push(current);
            seen.add(current);
          }
        }
        current = '';
      }
    }
    if (current && current.length > 2 && !this.stopwords.has(current) && !/^\d+$/.test(current)) {
      if (!seen.has(current)) words.push(current);
    }
    return words;
  }

  /**
   * Coherence via Cosine Similarity - optimized with pre-tokenized sentences
   */
  private calculateCoherence(sentences: string[]): number {
    if (sentences.length < 2) return sentences.length === 1 ? 70 : 25;

    let totalSimilarity = 0;
    let comparisons = 0;

    for (let i = 0; i < sentences.length - 1; i++) {
      const tokens1 = this.tokenize(sentences[i]);
      const tokens2 = this.tokenize(sentences[i + 1]);

      if (tokens1.length === 0 || tokens2.length === 0) continue;

      const set1 = new Set(tokens1);
      const set2 = new Set(tokens2);
      let intersection = 0;

      for (const token of set1) {
        if (set2.has(token)) intersection++;
      }

      const similarity = intersection / Math.sqrt(set1.size * set2.size);
      totalSimilarity += similarity;
      comparisons++;
    }

    if (comparisons === 0) return 40;

    const avg = totalSimilarity / comparisons;
    return avg < 0.2 ? 25 + (avg / 0.2) * 25 :
           avg < 0.5 ? 50 + ((avg - 0.2) / 0.3) * 25 :
           75 + ((avg - 0.5) / 0.5) * 20;
  }

  /**
   * TF-IDF - optimized with single pass
   */
  private calculateTFIDF(tokens: string[]): Map<string, number> {
    const tfIdf = new Map<string, number>();
    const freq = new Map<string, number>();

    for (const token of tokens) {
      freq.set(token, (freq.get(token) || 0) + 1);
    }

    const total = tokens.length;
    for (const [term, count] of freq) {
      const tf = count / total;
      const idf = this.vocabularyIDF.get(term) ?? Math.log((this.documentHistory.length + 1) / 1) + 1;
      tfIdf.set(term, tf * idf);
    }

    return tfIdf;
  }

  /**
   * Extract top N keywords - optimized sort
   */
  private extractTopKeywords(tfIdfScores: Map<string, number>, topN: number): string[] {
    const entries = Array.from(tfIdfScores.entries());
    entries.sort((a, b) => b[1] - a[1]);
    return entries.slice(0, topN).map(([word]) => word);
  }

  /**
   * Overall relevance - optimized
   */
  private calculateOverallRelevance(tfIdfScores: Map<string, number>, totalTokens: number): number {
    if (tfIdfScores.size === 0) return 30;

    const scores = Array.from(tfIdfScores.values()).sort((a, b) => b - a).slice(0, 8);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;

    return Math.min(100, avg * 60 + tfIdfScores.size * 1.5);
  }

  /**
   * Topic relevance - Jaccard + partial match
   */
  private calculateTopicRelevance(tokens: string[], topic: string): number {
    const topicTokens = this.tokenize(topic);
    if (topicTokens.length === 0) return 50;

    const transcriptSet = new Set(tokens);
    const topicSet = new Set(topicTokens);

    let intersection = 0;
    for (const t of topicSet) {
      if (transcriptSet.has(t)) intersection++;
    }

    const union = transcriptSet.size + topicSet.size - intersection;
    const jaccard = union === 0 ? 0 : intersection / union;

    let partial = 0;
    for (const t of topicSet) {
      if (tokens.some(token => token.includes(t) || t.includes(token))) partial++;
    }
    const partialScore = partial / topicSet.size;

    return (jaccard * 0.7 + partialScore * 0.3) * 100;
  }

  /**
   * VADER Sentiment - optimized word loop
   */
  private analyzeSentiment(text: string): { score: number; label: string } {
    const words = text.split(/\s+/);
    let score = 0;
    let modifier = 1;
    let negation = 0;

    for (let i = 0; i < words.length; i++) {
      const clean = words[i].toLowerCase().replace(/[^\w]/g, '');
      if (!clean) continue;

      const val = this.sentimentLexicon.get(clean);
      if (val === undefined) {
        if (negation > 0) negation--;
        continue;
      }

      if (Math.abs(val) < 2) {
        modifier = val;
        if (val < 0) negation = 3;
      } else {
        let adjusted = val * modifier;
        if (negation > 0) {
          adjusted *= -0.75;
          negation--;
        }
        score += adjusted;
        modifier = 1;
      }
    }

    const normalized = Math.max(-100, Math.min(100, score * 3));
    const label = normalized > 15 ? 'positive' : normalized < -15 ? 'negative' : 'neutral';
    return { score: normalized, label };
  }

  /**
   * NER - optimized regex + context
   */
  private extractEntities(text: string): string[] {
    const entities = new Set<string>();
    const words = text.split(/\s+/);
    let i = 0;

    while (i < words.length) {
      const word = words[i].replace(/[^\w]/g, '');
      if (!word) { i++; continue; }

      if (/^[A-Z][a-z]{2,}$/.test(word) && !this.isCommonWord(word)) {
        // Start of potential entity
        let entity = word;
        i++;
        while (i < words.length) {
          const next = words[i].replace(/[^\w]/g, '');
          if (/^[A-Z][a-z]+$/.test(next)) {
            entity += ' ' + next;
            i++;
          } else break;
        }
        entities.add(entity);
      } else {
        i++;
      }
    }

    return Array.from(entities).sort();
  }

  private isCommonWord(word: string): boolean {
    return ['The', 'This', 'That', 'These', 'Those', 'Then', 'There',
            'When', 'Where', 'Why', 'How', 'What', 'Which', 'Who'].includes(word);
  }

  /**
   * Vocabulary richness - TTR with logarithmic scaling
   */
  private calculateVocabularyRichness(tokens: string[]): number {
    if (tokens.length === 0) return 0;
    const unique = new Set(tokens).size;
    const ttr = unique / tokens.length;
    return Math.min(100, ttr * 140); // Scale to reach ~80 for rich speech
  }

  /**
   * Readability - optimized
   */
  private calculateReadability(text: string): number {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
    if (sentences.length === 0) return 50;

    const words = text.split(/\s+/).filter(w => w.length > 0);
    const avgSent = words.length / sentences.length;
    const avgWord = words.reduce((sum, w) => sum + w.length, 0) / words.length;

    const sentScore = Math.max(0, 100 - Math.abs(avgSent - 17.5) * 3.5);
    const wordScore = Math.max(0, 100 - Math.abs(avgWord - 5.2) * 12);

    return Math.round(sentScore * 0.65 + wordScore * 0.35);
  }

  /**
   * Update IDF with batch processing
   */
  private updateDocumentHistory(text: string): void {
    this.documentHistory.push(text);
    if (this.documentHistory.length > this.MAX_HISTORY) {
      this.documentHistory.shift();
    }
    this.updateIDF();
  }

  private updateIDF(): void {
    const docCount = this.documentHistory.length;
    if (docCount === 0) return;

    const termDocCount = new Map<string, number>();
    for (const doc of this.documentHistory) {
      const unique = new Set(this.tokenize(doc));
      for (const term of unique) {
        termDocCount.set(term, (termDocCount.get(term) || 0) + 1);
      }
    }

    for (const [term, count] of termDocCount) {
      const idf = Math.log((docCount + 1) / (count + 1)) + 1;
      this.vocabularyIDF.set(term, idf);
    }
  }

  private getDefaultMetrics(): ContentMetrics {
    return {
      coherenceScore: 0,
      keywordRelevance: 0,
      sentimentScore: 0,
      sentimentLabel: 'neutral',
      entityCount: 0,
      topKeywords: [],
      topEntities: [],
      vocabularyRichness: 0,
      readabilityScore: 0,
    };
  }

  reset(): void {
    this.documentHistory = [];
    this.vocabularyIDF.clear();
  }
}
