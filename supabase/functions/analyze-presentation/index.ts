import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { imageData, audioData, transcript } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');

    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY is not configured');
    }

    // Analyze facial expressions and body language from image
    const visionAnalysis = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          {
            role: 'system',
            content: `You are an expert presentation and speech therapy coach analyzing facial expressions, eye contact, posture, and body language from video frames.

IMPORTANT: Provide SPECIFIC, ACTIONABLE feedback based on what you actually observe in the image. Do not give generic advice.

Analyze the image carefully and provide scores (25-100) for:
- Eye Contact: Is the person looking directly at the camera? Check gaze direction. (25-100)
- Posture: Is their posture upright, shoulders back, confident? Check body alignment. (25-100)
- Facial Expression: Do they appear confident and engaged? Check for smiling, open expression, relaxed face. (25-100)
- Body Language: Are their gestures natural? Is their position stable? Check for nervous movements, hands visible, open stance. (25-100)

Be critical and honest. If something is wrong, score it lower and explain specifically what needs improvement.

Respond ONLY with valid JSON in this exact format:
{
  "eyeContact": <number 25-100>,
  "posture": <number 25-100>,
  "expression": <number 25-100>,
  "bodyLanguage": <number 25-100>,
  "feedback": "<specific feedback based on actual observation, mention specific issues seen>"
}`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze this presentation frame for eye contact, posture, facial expression, and body language. Be specific about what you observe.'
              },
              {
                type: 'image_url',
                image_url: {
                  url: imageData
                }
              }
            ]
          }
        ]
      })
    });

    const visionResult = await visionAnalysis.json();
    const visionScores = JSON.parse(visionResult.choices[0].message.content);

    // Analyze voice quality and speech content
    let voiceScores = { clarity: 70, pace: 70, tone: 70, engagement: 70, feedback: 'Not enough speech data yet.' };
    
    if (transcript && transcript.length > 20) {
      const voiceAnalysis = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${LOVABLE_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'google/gemini-2.5-flash',
          messages: [
            {
              role: 'system',
              content: `You are an expert speech therapist and presentation coach analyzing speech content and delivery for people with speaking difficulties.

IMPORTANT: Provide SPECIFIC, CONSTRUCTIVE feedback based on the actual transcript. Identify specific issues like:
- Filler words (um, uh, like, you know)
- Repetitions
- Incomplete sentences
- Pace issues (too fast/slow)
- Clarity issues (mumbling, unclear pronunciation)
- Confidence indicators (hesitations, qualifiers)

Analyze the transcript and provide scores (25-100) for:
- Clarity: Is the speech clear and articulate? Check for mumbling, slurring, unclear words. (25-100)
- Pace: Is the speaking pace appropriate (not too fast/slow)? Count words per minute if possible. (25-100)
- Tone: Is the tone engaging, professional, and confident? Check for monotone vs. varied. (25-100)
- Engagement: Is the content well-structured and engaging? Check for filler words, repetitions. (25-100)

Be honest and specific. Mention exact problems you see in the transcript.

Respond ONLY with valid JSON in this exact format:
{
  "clarity": <number 25-100>,
  "pace": <number 25-100>,
  "tone": <number 25-100>,
  "engagement": <number 25-100>,
  "feedback": "<specific feedback mentioning actual issues from the transcript>"
}`
            },
            {
              role: 'user',
              content: `Analyze this presentation transcript and provide specific, actionable feedback: "${transcript}"`
            }
          ]
        })
      });

      const voiceResult = await voiceAnalysis.json();
      voiceScores = JSON.parse(voiceResult.choices[0].message.content);
      console.log('Voice analysis completed:', voiceScores);
    }

    return new Response(
      JSON.stringify({
        vision: visionScores,
        voice: voiceScores,
        overall: Math.round(
          (visionScores.eyeContact + visionScores.posture + 
           voiceScores.clarity + voiceScores.engagement) / 4
        )
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Error in analyze-presentation function:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
