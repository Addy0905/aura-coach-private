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
            content: `You are an expert presentation coach analyzing facial expressions, eye contact, and body language.
Analyze the image and provide scores (25-100) for:
- Eye Contact: Are they looking at the camera? (25-100)
- Posture: Is their posture professional and confident? (25-100)
- Facial Expression: Do they appear engaged and confident? (25-100)
- Body Language: Are their gestures natural and appropriate? (25-100)

Respond ONLY with valid JSON in this exact format:
{
  "eyeContact": <number>,
  "posture": <number>,
  "expression": <number>,
  "bodyLanguage": <number>,
  "feedback": "<brief feedback>"
}`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze this presentation frame for eye contact, posture, facial expression, and body language.'
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
    let voiceScores = { clarity: 70, pace: 70, tone: 70, engagement: 70, feedback: '' };
    
    if (transcript && transcript.length > 10) {
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
              content: `You are an expert speech coach analyzing presentation content and delivery.
Analyze the transcript and provide scores (25-100) for:
- Clarity: Is the speech clear and articulate? (25-100)
- Pace: Is the speaking pace appropriate? (25-100)
- Tone: Is the tone engaging and professional? (25-100)
- Engagement: Is the content engaging and well-structured? (25-100)

Respond ONLY with valid JSON in this exact format:
{
  "clarity": <number>,
  "pace": <number>,
  "tone": <number>,
  "engagement": <number>,
  "feedback": "<brief feedback>"
}`
            },
            {
              role: 'user',
              content: `Analyze this presentation transcript: "${transcript}"`
            }
          ]
        })
      });

      const voiceResult = await voiceAnalysis.json();
      voiceScores = JSON.parse(voiceResult.choices[0].message.content);
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
