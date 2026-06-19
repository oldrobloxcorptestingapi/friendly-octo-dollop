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
//   nvidia_chat   — NVIDIA NIM chat completions   (env: NVIDIA_API_KEY)
//   groq_chat     — Groq chat completions         (env: GROQ_API_KEY)
//   github_chat   — GitHub Models completions     (env: GITHUB_TOKEN)
//   google_chat   — Google AI Studio completions  (env: GOOGLE_AI_KEY)
//   cerebras_chat — Cerebras chat completions     (env: CEREBRAS_API_KEY)
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
      case 'search':        return await handleSearch(params, res);
      case 'extract':       return await handleExtract(params, res);
      case 'nvidia_chat':   return await handleNvidiaChat(params, res);
      case 'groq_chat':     return await handleGroqChat(params, res);
      case 'github_chat':   return await handleGithubChat(params, res);
      case 'google_chat':   return await handleGoogleChat(params, res);
      case 'cerebras_chat': return await handleCerebrasChat(params, res);
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

// ── SHARED: OpenAI-compatible streaming chat ────────────────────────────
// All four new providers (Groq, GitHub Models, Google AI Studio, Cerebras)
// speak the OpenAI chat-completions wire format, so they share this handler.
//
// providerConfig:
//   url          — Full endpoint URL for /chat/completions
//   apiKey       — Resolved API key (from process.env.*).  Will return 503
//                  if falsy so the error is obvious during local dev.
//   providerName — Human-readable label used in error messages.
//
// Accepted params (same shape as nvidia_chat for consistency):
//   model, messages, tools, tool_choice, stream, max_tokens, temperature
//
// Note: the `thinking` param is intentionally NOT forwarded here — it is
// NVIDIA-NIM-specific and will cause a 400 on every other provider.
async function handleOpenAICompatibleChat(params, res, { url, apiKey, providerName }) {
  if (!apiKey) {
    return res.status(503).json({ error: `${providerName} API key not configured (check Vercel env vars)` });
  }

  const {
    model,
    messages,
    tools,
    tool_choice,
    stream      = true,
    max_tokens  = 4096,
    temperature = 0.7,
  } = params;

  if (!model) {
    return res.status(400).json({ error: 'Missing "model"' });
  }
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Missing or empty "messages" array' });
  }

  const body = { model, messages, stream, max_tokens, temperature };
  if (tools && tools.length) body.tools       = tools;
  if (tool_choice)           body.tool_choice = tool_choice;

  const r = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!r.ok) {
    const text = await r.text();
    return res.status(r.status).json({
      error: `${providerName} error (${r.status}): ${text.slice(0, 400)}`,
    });
  }

  if (!stream) {
    return res.status(200).json(await r.json());
  }

  // Streaming: normalize \r\n → \n so the frontend readStream() (which splits
  // on \n) can parse SSE events reliably regardless of provider line endings.
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

      const text = (leftover + decoder.decode(value, { stream: true }))
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n');

      const parts = text.split('\n\n');
      leftover = parts.pop(); // hold back any incomplete trailing event

      for (const part of parts) {
        res.write(part + '\n\n');
      }
    }
  } catch (e) {
    console.error(`[${providerName}] stream error:`, e);
  } finally {
    res.end();
  }
}

// ── GROQ chat completions ───────────────────────────────────────────────
// https://console.groq.com/docs/openai
// Env var: GROQ_API_KEY
// Groq runs open-source models (Llama, DeepSeek, Mistral, Qwen…) on their
// custom LPU hardware at extremely high token throughput.
async function handleGroqChat(params, res) {
  return handleOpenAICompatibleChat(params, res, {
    url:          'https://api.groq.com/openai/v1/chat/completions',
    apiKey:       process.env.GROQ_API_KEY,
    providerName: 'Groq',
  });
}

// ── GITHUB MODELS chat completions ─────────────────────────────────────
// https://docs.github.com/en/github-models
// Env var: GITHUB_TOKEN  (a fine-grained PAT with "models:read" scope, or a
//          classic token — GitHub Models currently accepts both.)
// Hosts GPT-4.1, o-series, Llama, Phi, Mistral, and more on Azure infra.
async function handleGithubChat(params, res) {
  return handleOpenAICompatibleChat(params, res, {
    url:          'https://models.inference.ai.azure.com/chat/completions',
    apiKey:       process.env.GITHUB_TOKEN,
    providerName: 'GitHub Models',
  });
}

// ── GOOGLE AI STUDIO chat completions ──────────────────────────────────
// https://ai.google.dev/gemini-api/docs/openai
// Env var: GOOGLE_AI_KEY  (API key from https://aistudio.google.com/apikey)
// Uses Google's OpenAI-compatible shim so the same wire format works for
// all Gemini models (gemini-2.5-pro, gemini-2.5-flash, etc.).
async function handleGoogleChat(params, res) {
  return handleOpenAICompatibleChat(params, res, {
    url:          'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    apiKey:       process.env.GOOGLE_AI_KEY,
    providerName: 'Google AI Studio',
  });
}

// ── CEREBRAS chat completions ───────────────────────────────────────────
// https://inference-docs.cerebras.ai/introduction
// Env var: CEREBRAS_API_KEY
// Cerebras runs models on wafer-scale chips delivering ~2,000 token/s for
// large models (llama-3.3-70b) — the fastest publicly available inference.
async function handleCerebrasChat(params, res) {
  return handleOpenAICompatibleChat(params, res, {
    url:          'https://api.cerebras.ai/v1/chat/completions',
    apiKey:       process.env.CEREBRAS_API_KEY,
    providerName: 'Cerebras',
  });
}

// ── NVIDIA NIM: chat completions ────────────────────────────────────────
// https://docs.api.nvidia.com/nim/reference/llm-apis
//
// Kept as a dedicated handler (rather than folded into handleOpenAICompatibleChat)
// because NIM has two unique requirements:
//   1. The `thinking` extension param (budget_tokens) is NIM-specific.
//   2. NIM is stricter about message shape (content must never be null/undefined).
//
// Env var: NVIDIA_API_KEY
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

  if (!process.env.NVIDIA_API_KEY) {
    return res.status(503).json({ error: 'NVIDIA API key not configured (check Vercel env vars)' });
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

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');

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

      const text = (leftover + decoder.decode(value, { stream: true }))
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n');

      const parts = text.split('\n\n');
      leftover = parts.pop();

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
