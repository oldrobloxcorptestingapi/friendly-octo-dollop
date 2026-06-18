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
      case 'search':  return await handleSearch(params, res);
      case 'extract': return await handleExtract(params, res);

      // ── Add new actions here as you wire up more services ──
      // case 'image':  return await handleImage(params, res);
      // case 'tts':    return await handleTTS(params, res);
      // case 'chat':   return await handleDirectChat(params, res);

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
