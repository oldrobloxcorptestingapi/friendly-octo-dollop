// ──────────────────────────────────────────────────────────────────────────
// Traz backend — single entry point for server-side AI service calls.
// Deployed as one Vercel serverless function at POST /api/main.
//
// Why this exists: anything that needs a secret key kept out of the
// browser (or that browsers can't call directly) goes through here instead
// of being called straight from index.html.
//
// Request shape:  POST { action: '<name>', ...params }
// Response shape: 200 { ...actionResult }  |  4xx/5xx { error: '...' }
//
// Actions:
//   search        — Tavily web search
//   extract       — Tavily page extraction
//   nvidia_chat   — NVIDIA NIM chat completions (streaming SSE pass-through)
//
// Add more services later by adding a new case to the switch below and a
// handler function. Keep each handler self-contained so this file can grow
// without the actions stepping on each other.
// ──────────────────────────────────────────────────────────────────────────

module.exports = async (req, res) => {
  // CORS — open for now since the key never leaves the server.
  // Tighten this to your actual frontend origin once Traz has a fixed domain.
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Use POST' });

  const { action, ...params } = req.body || {};

  try {
    switch (action) {
      case 'search':      return await handleSearch(params, res);
      case 'extract':     return await handleExtract(params, res);
      case 'nvidia_chat': return await handleNvidiaChat(params, res);

      // ── Add new actions here as you wire up more services ──
      // case 'image':  return await handleImage(params, res);
      // case 'tts':    return await handleTTS(params, res);

      default:
        return res.status(400).json({ error: `Unknown action: "${action}"` });
    }
  } catch (e) {
    console.error(`[${action}]`, e);
    return res.status(500).json({ error: e.message || 'Internal error' });
  }
};

// ── TAVILY: web search ──────────────────────────────────────────────────
// https://docs.tavily.com/documentation/api-reference/endpoint/search
async function handleSearch({ query, max_results = 5, search_depth = 'basic' }, res) {
  if (!query) return res.status(400).json({ error: 'Missing "query"' });

  const r = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.TAVILY_API_KEY}`,
    },
    body: JSON.stringify({
      query,
      max_results,
      search_depth,        // 'basic' (1 credit) or 'advanced' (2 credits)
      include_answer: true, // Tavily's own synthesized answer, handy for the bubble
    }),
  });

  if (!r.ok) {
    const text = await r.text();
    return res.status(r.status).json({ error: `Tavily search failed: ${text.slice(0, 300)}` });
  }
  return res.status(200).json(await r.json());
}

// ── TAVILY: page extraction ─────────────────────────────────────────────
// https://docs.tavily.com/documentation/api-reference/endpoint/extract
async function handleExtract({ url }, res) {
  if (!url) return res.status(400).json({ error: 'Missing "url"' });

  const r = await fetch('https://api.tavily.com/extract', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.TAVILY_API_KEY}`,
    },
    body: JSON.stringify({ urls: [url] }),
  });

  if (!r.ok) {
    const text = await r.text();
    return res.status(r.status).json({ error: `Tavily extract failed: ${text.slice(0, 300)}` });
  }
  return res.status(200).json(await r.json());
}

// ── NVIDIA NIM: chat completions ────────────────────────────────────────
// https://docs.api.nvidia.com/nim/reference/llm-apis
//
// Proxies the full OpenAI-compatible /chat/completions request to NVIDIA's
// NIM API, forwarding streaming SSE back to the client so the frontend
// readStream() function works without modification.
//
// Required env var: NVIDIA_API_KEY  (set in Vercel → Settings → Environment)
// Supported models (pass as `model` in the request body):
//   nvidia/llama-3.1-nemotron-ultra-253b-v1          — flagship reasoning
//   nvidia/llama-3.1-nemotron-70b-instruct            — balanced
//   nvidia/llama-3.3-nemotron-super-49b-v1            — fast + smart
//   meta/llama-3.1-405b-instruct                      — Meta's largest
//   mistralai/mixtral-8x22b-instruct-v0.1             — MoE option
//
// The frontend sends:  POST { action:'nvidia_chat', model, messages, tools?,
//                              tool_choice?, stream?, max_tokens?, temperature? }
async function handleNvidiaChat(params, res) {
  const {
    model    = 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    messages,
    tools,
    tool_choice,
    stream      = true,
    max_tokens  = 4096,
    temperature = 0.7,
    thinking,        // extended-thinking hint, forwarded if present
  } = params;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Missing or empty "messages" array' });
  }

  const body = {
    model,
    messages,
    stream,
    max_tokens,
    temperature,
  };
  if (tools && tools.length)    body.tools        = tools;
  if (tool_choice)              body.tool_choice  = tool_choice;
  if (thinking)                 body.thinking     = thinking;

  const r = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${process.env.NVIDIA_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!r.ok) {
    const text = await r.text();
    return res.status(r.status).json({ error: `NVIDIA API error: ${text.slice(0, 400)}` });
  }

  if (!stream) {
    // Non-streaming: just proxy the JSON response
    return res.status(200).json(await r.json());
  }

  // Streaming: pipe SSE chunks straight back to the client.
  // The frontend's readStream() already speaks this format.
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no'); // disable Nginx buffering on Vercel

  const reader = r.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(decoder.decode(value, { stream: true }));
    }
  } catch (e) {
    console.error('[nvidia_chat] stream error:', e);
  } finally {
    res.end();
  }
}
