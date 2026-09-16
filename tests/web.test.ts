import { test } from 'node:test';
import assert from 'node:assert';
import { parseDuckDuckGoHtml, unescapeHtml, unwrapDdgUrl } from '../src/tools/web_search.js';
import { htmlToMarkdown, webFetchTool } from '../src/tools/web_fetch.js';
import { isBlockedSearchPage } from '../src/tools/web_search.js';
import { networkTimeoutMs } from '../src/tools/network.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { Agent } from '../src/agent.js';
import { DeepSeekClient } from '../src/llm/client.js';

test('Web tools: DuckDuckGo HTML parser and entity decoding', async (t) => {
  await t.test('unescapeHtml handles entities and strips inner tags', () => {
    const raw = '<b>Python 3.14</b> &amp; &quot;PEP 649&#x27;s&quot; deferred &lt;evaluation&gt;';
    assert.strictEqual(unescapeHtml(raw), 'Python 3.14 & "PEP 649\'s" deferred <evaluation>');
  });

  await t.test('unwrapDdgUrl decodes uddg query parameter', () => {
    const ddgLink = '//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3.14%2Fwhatsnew&rut=123';
    const unwrapped = unwrapDdgUrl(ddgLink);
    assert.strictEqual(unwrapped, 'https://docs.python.org/3.14/whatsnew');
  });

  await t.test('parseDuckDuckGoHtml extracts results cleanly', () => {
    const mockHtml = `
      <div class="results">
        <div class="result results_links results_links_deep web-result ">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fanthropic.com%2Fnews%2Fclaude-3-7-sonnet">Claude 3.7 Sonnet &amp; Hybrid Reasoning</a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fanthropic.com%2Fnews%2Fclaude-3-7-sonnet">
            Today we are announcing <b>Claude 3.7 Sonnet</b>, our most intelligent model to date...
          </a>
        </div>
        <div class="result results_links results_links_deep web-result ">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3.14">What&#39;s New In Python 3.14</a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3.14">
            Python 3.14 includes deferred evaluation of annotations and new CLI features.
          </a>
        </div>
      </div>
    `;

    const results = parseDuckDuckGoHtml(mockHtml, 5);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0].title, 'Claude 3.7 Sonnet & Hybrid Reasoning');
    assert.strictEqual(results[0].url, 'https://anthropic.com/news/claude-3-7-sonnet');
    assert.match(results[0].snippet, /Today we are announcing Claude 3\.7 Sonnet/);

    assert.strictEqual(results[1].title, "What's New In Python 3.14");
    assert.strictEqual(results[1].url, 'https://docs.python.org/3.14');
    assert.match(results[1].snippet, /Python 3\.14 includes deferred evaluation/);
  });
});

test('Web tools: HTML to Markdown and security checks', async (t) => {
  await t.test('non-http schemes are still refused', async () => {
    for (const url of ['file:///etc/passwd', 'ftp://example.com/x', 'data:text/html,hi']) {
      assert.match(await webFetchTool.execute({ url }, { cwd: '/tmp' }), /only http: and https:/);
    }
  });

  await t.test('htmlToMarkdown strips boilerplate and converts formatting', () => {
    const mockPageHtml = `
      <!DOCTYPE html>
      <html>
        <head><title>Docs</title><style>.hidden{display:none;}</style></head>
        <body>
          <header><nav><a href="/home">Home</a></nav></header>
          <main>
            <h1>Python 3.14 Features</h1>
            <p>Here is an introduction to deferred evaluation.</p>
            <h2>Code Example</h2>
            <pre><code class="python">def compute(x: int) -> int:
    return x * 2</code></pre>
            <p>Read more at <a href="https://peps.python.org/pep-0649/">PEP 649</a>.</p>
            <ul>
              <li>Item 1</li>
              <li>Item 2</li>
            </ul>
          </main>
          <footer><p>Copyright 2026</p></footer>
          <script>console.log('tracker');</script>
        </body>
      </html>
    `;

    const md = htmlToMarkdown(mockPageHtml, 5000);
    assert.doesNotMatch(md, /<style/);
    assert.doesNotMatch(md, /<script/);
    assert.doesNotMatch(md, /tracker/);
    assert.doesNotMatch(md, /Home/);
    assert.doesNotMatch(md, /Copyright 2026/);

    assert.match(md, /# Python 3\.14 Features/);
    assert.match(md, /## Code Example/);
    assert.match(md, /```\ndef compute\(x: int\) -> int:\n {4}return x \* 2\n```/);
    assert.match(md, /\[PEP 649\]\(https:\/\/peps\.python\.org\/pep-0649\/\)/);
    assert.match(md, /- Item 1/);
  });
});

test('ToolRegistry and Agent mode integration for web tools', async () => {
  const registry = new ToolRegistry();
  const tools = registry.getOpenAITools();
  const toolNames = tools.map((t) => t.function.name);

  assert.ok(toolNames.includes('web_search'), 'ToolRegistry must include web_search');
  assert.ok(toolNames.includes('web_fetch'), 'ToolRegistry must include web_fetch');

  // Verify plan mode includes web_search and web_fetch
  const client = new DeepSeekClient();
  let receivedTools: any[] = [];
  client.streamChat = async (_msgs, passedTools) => {
    receivedTools = passedTools;
    return { content: 'OK', toolCalls: [] };
  };

  const agent = new Agent({ client });
  agent.setInteractionMode('plan');
  await agent.runTurn('Research python 3.14', {});

  const planToolNames = receivedTools.map((t) => t.function.name);
  assert.ok(planToolNames.includes('web_search'), 'Plan mode must permit web_search');
  assert.ok(planToolNames.includes('web_fetch'), 'Plan mode must permit web_fetch');
  assert.ok(planToolNames.includes('view_file'), 'Plan mode must permit view_file');
  assert.strictEqual(planToolNames.includes('write_file'), false, 'Plan mode must NOT permit write_file');
  assert.strictEqual(planToolNames.includes('bash'), false, 'Plan mode must NOT permit bash');
});

test('Markdown terminal rendering produces clickable OSC 8 hyperlinks', async () => {
  const { Markdown, getCapabilities } = await import('@earendil-works/pi-tui');
  const { markdownTheme } = await import('../src/tui/theme.js');

  assert.strictEqual(getCapabilities().hyperlinks, true, 'Terminal hyperlinks capability must be enabled');

  const md = new Markdown('Check [Anthropic Blog](https://anthropic.com/news) for updates', 0, 0, markdownTheme);
  const rendered = md.render(80).join('\n');

  // OSC 8 opening sequence: \x1b]8;;url\x1b\
  assert.ok(rendered.includes('\x1b]8;;https://anthropic.com/news\x1b\\'), 'Rendered line must contain OSC 8 URL target');
  // Link text should be present and styled
  assert.ok(rendered.includes('Anthropic Blog'), 'Rendered line must contain visible link text');
  // OSC 8 closing sequence: \x1b]8;;\x1b\
  assert.ok(rendered.includes('\x1b]8;;\x1b\\'), 'Rendered line must contain OSC 8 closing delimiter');
});


test('A rate-limited search is not reported as an empty web', async (t) => {
  const blockPage = '<html><head><link rel="canonical" href="https://duckduckgo.com/"></head><body>anomaly</body></html>';
  const resultsPage = '<a class="result__a" href="https://example.com">Title</a><a class="result__snippet">Snip</a>';

  await t.test('DuckDuckGo answers 202 with a block page, not an error status', () => {
    assert.strictEqual(isBlockedSearchPage(202, blockPage), true);
  });

  await t.test('a page carrying results is never treated as blocked', () => {
    assert.strictEqual(isBlockedSearchPage(200, resultsPage), false);
    assert.strictEqual(isBlockedSearchPage(202, resultsPage), false, 'results present outrank the status');
  });

  await t.test('a genuinely empty result page is not mistaken for rate limiting', () => {
    assert.strictEqual(isBlockedSearchPage(200, '<html><body>No results for that query.</body></html>'), false);
  });
});

test('Network deadlines can be raised for a slow link', async () => {
  assert.strictEqual(networkTimeoutMs('45000', 30_000), 45_000);
  assert.strictEqual(networkTimeoutMs(undefined, 30_000), 30_000);
  assert.strictEqual(networkTimeoutMs('not-a-number', 30_000), 30_000);
  assert.strictEqual(networkTimeoutMs('0', 30_000), 30_000, 'zero would disable the deadline entirely');
});
