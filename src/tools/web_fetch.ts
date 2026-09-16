import type { ToolDefinition } from './types.js';
import { unescapeHtml } from './web_search.js';
import { networkTimeoutMs } from './network.js';

/**
 * How long a page has to answer. Measured fetches of large documentation pages land well under a
 * second; this is sized for a bad link, not a typical one, and only ever bounds the HTTP request —
 * never the model's own reading or reasoning, which happen after the tool returns.
 */
const REQUEST_TIMEOUT_MS = networkTimeoutMs(process.env.DSH_WEB_TIMEOUT_MS, 30_000);

/**
 * Reads at most `maxBytes` of the body. `res.text()` would buffer the whole response before the
 * length cap is applied, so a huge or endless page would cost memory for content that is discarded.
 */
async function readBounded(res: Response, maxBytes: number): Promise<string> {
  if (!res.body) return '';
  const decoder = new TextDecoder();
  const reader = res.body.getReader();
  let text = '';
  let read = 0;
  try {
    while (read < maxBytes) {
      const { done, value } = await reader.read();
      if (done) break;
      read += value.byteLength;
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    await reader.cancel().catch((err) => void err);
  }
  return text;
}

/** Combines the caller's cancellation with a request deadline. */
function requestSignal(abortSignal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  return abortSignal ? AbortSignal.any([abortSignal, timeout]) : timeout;
}

export function htmlToMarkdown(html: string, maxLength = 8000): string {
  // 1. Remove non-content tags
  let text = html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
    .replace(/<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>/gi, '')
    .replace(/<svg\b[^<]*(?:(?!<\/svg>)<[^<]*)*<\/svg>/gi, '')
    .replace(/<nav\b[^<]*(?:(?!<\/nav>)<[^<]*)*<\/nav>/gi, '')
    .replace(/<header\b[^<]*(?:(?!<\/header>)<[^<]*)*<\/header>/gi, '')
    .replace(/<footer\b[^<]*(?:(?!<\/footer>)<[^<]*)*<\/footer>/gi, '')
    .replace(/<aside\b[^<]*(?:(?!<\/aside>)<[^<]*)*<\/aside>/gi, '');

  // 2. Extract main/article content if present
  const mainMatch = /<main\b[^>]*>([\s\S]*?)<\/main>/i.exec(text);
  const articleMatch = /<article\b[^>]*>([\s\S]*?)<\/article>/i.exec(text);
  if (articleMatch) {
    text = articleMatch[1];
  } else if (mainMatch) {
    text = mainMatch[1];
  } else {
    const bodyMatch = /<body\b[^>]*>([\s\S]*?)<\/body>/i.exec(text);
    if (bodyMatch) text = bodyMatch[1];
  }

  // 3. Convert code blocks before other tags
  text = text.replace(/<pre\b[^>]*><code\b[^>]*>([\s\S]*?)<\/code><\/pre>/gi, (_, code) => {
    return `\n\`\`\`\n${unescapeHtml(code)}\n\`\`\`\n`;
  });
  text = text.replace(/<code\b[^>]*>([\s\S]*?)<\/code>/gi, (_, code) => {
    return ` \`${unescapeHtml(code)}\` `;
  });

  // 4. Headings
  text = text.replace(/<h1\b[^>]*>([\s\S]*?)<\/h1>/gi, '\n# $1\n');
  text = text.replace(/<h2\b[^>]*>([\s\S]*?)<\/h2>/gi, '\n## $1\n');
  text = text.replace(/<h3\b[^>]*>([\s\S]*?)<\/h3>/gi, '\n### $1\n');
  text = text.replace(/<h[4-6]\b[^>]*>([\s\S]*?)<\/h[4-6]>/gi, '\n#### $1\n');

  // 5. Links, lists, formatting
  text = text.replace(/<a\b[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi, (_, href, content) => {
    const cleanContent = content.replace(/<[^>]+>/g, '').trim();
    if (!cleanContent) return '';
    return `[${cleanContent}](${href})`;
  });
  text = text.replace(/<li\b[^>]*>([\s\S]*?)<\/li>/gi, '\n- $1');
  text = text.replace(/<p\b[^>]*>([\s\S]*?)<\/p>/gi, '\n\n$1\n\n');
  text = text.replace(/<br\s*\/?>/gi, '\n');
  text = text.replace(/<hr\s*\/?>/gi, '\n---\n');

  // 6. Strip all remaining HTML tags
  text = text.replace(/<[^>]+>/g, '');

  // 7. Unescape entities and normalize blank lines
  text = unescapeHtml(text);
  const lines = text.split('\n').map((l) => l.trimEnd());
  const cleanLines: string[] = [];
  let prevBlank = false;
  for (const line of lines) {
    const isBlank = line.trim().length === 0;
    if (isBlank && prevBlank) continue;
    cleanLines.push(line);
    prevBlank = isBlank;
  }

  text = cleanLines.join('\n').trim();

  // 8. Enforce context length ceiling
  if (text.length > maxLength) {
    text = text.slice(0, maxLength) + `\n\n[... content truncated to preserve context budget (${maxLength} characters) ...]`;
  }

  return text;
}

export const webFetchTool: ToolDefinition<{ url: string; max_length?: number }, string> = {
  name: 'web_fetch',
  description: 'Fetch and read the main content of a web page as clean Markdown. Strips scripts, navigation, and ads.',
  parameters: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        description: 'The HTTP or HTTPS URL of the web page to fetch.',
      },
      max_length: {
        type: 'number',
        description: 'Maximum characters of content to return (default 8000, max 25000).',
      },
    },
    required: ['url'],
  },
  execute: async (args, context) => {
    const rawUrl = args.url?.trim();
    if (!rawUrl) {
      return 'Error: url is required.';
    }

    let parsed: URL;
    try {
      parsed = new URL(rawUrl);
    } catch {
      return `Error: invalid URL "${rawUrl}".`;
    }

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return `Error: only http: and https: protocols are supported.`;
    }

    context.onProgress?.(`Fetching: ${rawUrl}...`);

    try {
      const res = await fetch(rawUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.9',
        },
        redirect: 'follow',
        signal: requestSignal(context.abortSignal),
      });

      if (!res.ok) {
        return `Error fetching "${rawUrl}": HTTP ${res.status} ${res.statusText}`;
      }

      const contentType = res.headers.get('content-type') || '';
      const maxLength = Math.min(Math.max(args.max_length || 8000, 1000), 25000);
      // Markup costs far more bytes than the text it yields, so read well past the text budget.
      const bodyText = await readBounded(res, maxLength * 20);

      // If already markdown or plain text, return directly
      if (contentType.includes('text/plain') || contentType.includes('text/markdown') || rawUrl.endsWith('.md')) {
        const truncated = bodyText.length > maxLength
          ? bodyText.slice(0, maxLength) + `\n\n[... truncated to ${maxLength} characters ...]`
          : bodyText;
        return `Content from ${rawUrl}:\n\n${truncated}`;
      }

      const markdown = htmlToMarkdown(bodyText, maxLength);
      if (!markdown) {
        return `Fetched ${rawUrl}, but no readable text content was found.`;
      }

      return `Content from ${rawUrl}:\n\n${markdown}`;
    } catch (err: any) {
      if (context.abortSignal?.aborted) {
        return 'Web fetch aborted by user.';
      }
      if (err?.name === 'TimeoutError') {
        return `Error fetching "${rawUrl}": no response within ${REQUEST_TIMEOUT_MS / 1000}s.`;
      }
      return `Error fetching "${rawUrl}": ${err.message}`;
    }
  },
};
