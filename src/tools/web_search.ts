import type { ToolDefinition } from './types.js';
import { networkTimeoutMs } from './network.js';

/**
 * How long a search backend has to answer, per attempt. Bounds the HTTP request only; the model's
 * reading of the results is not on this clock.
 */
const SEARCH_TIMEOUT_MS = networkTimeoutMs(process.env.DSH_WEB_TIMEOUT_MS, 20_000);

function searchSignal(abortSignal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(SEARCH_TIMEOUT_MS);
  return abortSignal ? AbortSignal.any([abortSignal, timeout]) : timeout;
}

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

export function unescapeHtml(str: string): string {
  return str
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .trim();
}

export function unwrapDdgUrl(rawUrl: string): string {
  try {
    const full = rawUrl.startsWith('//') ? `https:${rawUrl}` : rawUrl;
    const parsed = new URL(full, 'https://duckduckgo.com');
    const uddg = parsed.searchParams.get('uddg');
    if (uddg) {
      return decodeURIComponent(uddg);
    }
    return full;
  } catch {
    return rawUrl;
  }
}

export function parseDuckDuckGoHtml(html: string, limit = 5): SearchResult[] {
  const results: SearchResult[] = [];

  // Match each result block
  // DuckDuckGo HTML layout typically has:
  // <a class="result__a" href="...">Title</a> ... <a class="result__snippet" ...>Snippet</a>
  const resultRegex =
    /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/gi;

  let match: RegExpExecArray | null;
  while ((match = resultRegex.exec(html)) !== null) {
    const rawUrl = match[1];
    const rawTitle = match[2];
    const rawSnippet = match[3];

    const url = unwrapDdgUrl(rawUrl);
    const title = unescapeHtml(rawTitle);
    const snippet = unescapeHtml(rawSnippet);

    if (url && title && !url.includes('duckduckgo.com/y.js')) {
      results.push({ title, url, snippet });
      if (results.length >= limit) break;
    }
  }

  // Fallback pattern if the snippet is structured differently
  if (results.length === 0) {
    const titleRegex = /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
    while ((match = titleRegex.exec(html)) !== null) {
      const url = unwrapDdgUrl(match[1]);
      const title = unescapeHtml(match[2]);
      if (url && title && !url.includes('duckduckgo.com/y.js')) {
        results.push({ title, url, snippet: '' });
        if (results.length >= limit) break;
      }
    }
  }

  return results;
}

/**
 * Whether DuckDuckGo served its rate-limit page instead of results. It answers 200 or 202 either
 * way, so status alone cannot tell them apart, and an unrecognised page parses to zero results —
 * which would otherwise be reported as "nothing exists on the web about this".
 */
export function isBlockedSearchPage(status: number, html: string): boolean {
  if (/class="[^"]*result__a/.test(html)) return false;
  return status === 202 || /anomaly|unusual traffic|captcha|blocked/i.test(html);
}

async function fetchDuckDuckGoHtml(query: string, signal?: AbortSignal): Promise<{ status: number; html: string }> {
  const endpoint = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  const res = await fetch(endpoint, {
    method: 'GET',
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    },
    signal: searchSignal(signal),
  });

  if (!res.ok) {
    throw new Error(`DuckDuckGo search failed with status ${res.status} ${res.statusText}`);
  }

  return { status: res.status, html: await res.text() };
}

async function searchDuckDuckGo(query: string, limit: number, signal?: AbortSignal): Promise<SearchResult[]> {
  let attempt = await fetchDuckDuckGoHtml(query, signal);

  // Rate limiting here is transient, and the deadline leaves room for one more try.
  if (isBlockedSearchPage(attempt.status, attempt.html)) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    signal?.throwIfAborted();
    attempt = await fetchDuckDuckGoHtml(query, signal);
  }

  if (isBlockedSearchPage(attempt.status, attempt.html)) {
    throw new Error(
      'DuckDuckGo is rate-limiting this machine, so no results could be read. Retry shortly, set TAVILY_API_KEY or BRAVE_API_KEY for a keyed backend, or fetch a known URL with web_fetch.'
    );
  }

  return parseDuckDuckGoHtml(attempt.html, limit);
}

async function searchTavily(
  apiKey: string,
  query: string,
  limit: number,
  signal?: AbortSignal,
): Promise<SearchResult[]> {
  // Tavily authenticates with a bearer token; the api_key body field it once accepted is not in
  // the current API. search_depth is left at its default, which costs one credit per search.
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ query, max_results: limit }),
    signal: searchSignal(signal),
  });
  if (res.status === 401) {
    throw new Error('Tavily rejected TAVILY_API_KEY. Check the key, or unset it to fall back to DuckDuckGo.');
  }
  if (!res.ok) throw new Error(`Tavily search failed (${res.status})`);
  const data = (await res.json()) as any;
  return (data.results || []).map((r: any) => ({
    title: r.title || '',
    url: r.url || '',
    snippet: r.content || '',
  }));
}

async function searchBrave(
  apiKey: string,
  query: string,
  limit: number,
  signal?: AbortSignal,
): Promise<SearchResult[]> {
  const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${limit}`;
  const res = await fetch(url, {
    headers: { 'X-Subscription-Token': apiKey, Accept: 'application/json' },
    signal: searchSignal(signal),
  });
  if (!res.ok) throw new Error(`Brave search failed (${res.status})`);
  const data = (await res.json()) as any;
  return (data.web?.results || []).map((r: any) => ({
    title: r.title || '',
    url: r.url || '',
    snippet: r.description || '',
  }));
}

export const webSearchTool: ToolDefinition<{ query: string; limit?: number }, string> = {
  name: 'web_search',
  description:
    'Search the web using DuckDuckGo. Returns a list of titles, URLs, and relevant snippets for the query.',
  parameters: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'The search query to look up on the web.',
      },
      limit: {
        type: 'number',
        description: 'Maximum number of search results to return (default 5, max 10).',
      },
    },
    required: ['query'],
  },
  execute: async (args, context) => {
    const query = args.query?.trim();
    if (!query) {
      return 'Error: query is required.';
    }

    const limit = Math.min(Math.max(args.limit || 5, 1), 10);
    context.onProgress?.(`Searching web for: "${query}"...`);

    let results: SearchResult[] = [];
    try {
      if (process.env.TAVILY_API_KEY) {
        results = await searchTavily(process.env.TAVILY_API_KEY, query, limit, context.abortSignal);
      } else if (process.env.BRAVE_API_KEY) {
        results = await searchBrave(process.env.BRAVE_API_KEY, query, limit, context.abortSignal);
      } else {
        results = await searchDuckDuckGo(query, limit, context.abortSignal);
      }
    } catch (err: any) {
      if (context.abortSignal?.aborted) return 'Web search aborted by user.';
      if (err?.name === 'TimeoutError') {
        return `Web search timed out after ${SEARCH_TIMEOUT_MS / 1000}s.`;
      }
      return `Web search failed: ${err.message}`;
    }

    if (results.length === 0) {
      return `No web results found for "${query}".`;
    }

    const formatted = results
      .map(
        (r, i) => `${i + 1}. [${r.title}](${r.url})\n   ${r.snippet ? r.snippet : 'No snippet available.'}`,
      )
      .join('\n\n');

    return `Web search results for "${query}":\n\n${formatted}`;
  },
};
