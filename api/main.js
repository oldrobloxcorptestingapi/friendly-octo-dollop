// ──────────────────────────────────────────────────────────────────────────
// Traz backend — single entry point for server-side AI service calls.
// Deployed as one Vercel serverless function at POST /api/main.
//
// Request shape:  POST { action: '<name>', ...params }
// Response shape: 200 { ...actionResult }  |  4xx/5xx { error: '...' }
//
// Actions:
//   search        — Tavily web search
//   extract       — Tavily page extraction
//   nvidia_chat   — NVIDIA NIM chat completions (streaming SSE pass-through)
// ──────────────────────────────────────────────────────────────────────────

module.exports = async (req, res) => {
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

      // ── Add new actions here ──
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
      search_depth,
      include_answer: true,
    }),
  });

  if (!r.ok) {
    const text = await r.text();
    return res.status(r.status).json({ error: `Tavily search failed: ${text.slice(0, 300)}` });
  }
  return res.status(200).json(await r.json());
}

// ── TAVILY: page extraction ─────────────────────────────────────────────
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
// Proxies OpenAI-compatible /chat/completions to NVIDIA's NIM API.
// Normalizes \r\n line endings before forwarding SSE to the client so
// the frontend readStream() (which splits on \n) works correctly.
//
// Required env var: NVIDIA_API_KEY  (Vercel → Settings → Environment Variables)
async function handleNvidiaChat(params, res) {
  const {
    model       = 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    messages,
    tools,
    tool_choice,
    stream      = true,
    max_tokens  = 4096,
    temperature = 0.7,
    thinking,
  } = params;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Missing or empty "messages" array' });
  }

  const body = { model, messages, stream, max_tokens, temperature };
  if (tools && tools.length) body.tools       = tools;
  if (tool_choice)           body.tool_choice = tool_choice;
  if (thinking)              body.thinking    = thinking;

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
    return res.status(200).json(await r.json());
  }

  // Streaming: read NVIDIA's SSE, normalize \r\n → \n, forward complete events.
  // NVIDIA NIM emits \r\n line endings; the frontend splits on \n so we must
  // normalize or each line arrives as "data: {...}\r" and startsWith('data: ')
  // still matches but JSON.parse fails on the trailing \r.
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no'); // disable Nginx buffering on Vercel

  const reader  = r.body.getReader();
  const decoder = new TextDecoder();
  let leftover  = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        if (leftover.trim()) res.write(leftover + '\n');
        break;
      }

      // Normalize \r\n and stray \r to \n
      const text = (leftover + decoder.decode(value, { stream: true }))
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n');

      // Forward complete SSE events (double-newline separated); keep any
      // partial event in leftover so it's not forwarded mid-chunk.
      const parts = text.split('\n\n');
      leftover = parts.pop(); // last element may be an incomplete event

      for (const part of parts) {
        res.write(part + '\n\n');
      }
    }
  } catch (e) {
    console.error('[nvidia_chat] stream error:', e);
  } finally {
    res.end();
  }
}
