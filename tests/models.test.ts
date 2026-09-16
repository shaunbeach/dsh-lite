import { test } from 'node:test';
import assert from 'node:assert';
import * as fs from 'node:fs/promises';
import * as fsSync from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { effectiveServerContext, findModelsConfigPath, loadModelsConfig, resolveHomePath } from '../src/config/models.js';
import { ContextManager, elideToolResultContent, estimateMessageTokens } from '../src/context.js';
import { findFatalOutput, LlamaServerManager, sameLaunch, serverOrigin } from '../src/llm/server.js';
import type { ChatMessage } from '../src/llm/types.js';
import { DEFAULT_SAMPLING, DeepSeekClient } from '../src/llm/client.js';
import { Agent } from '../src/agent.js';

test('Models config and local llama.cpp infrastructure', async (t) => {
  await t.test('serverOrigin parses host origin', () => {
    assert.strictEqual(serverOrigin('http://localhost:8080/v1'), 'http://localhost:8080');
    assert.strictEqual(serverOrigin('http://127.0.0.1:9090/v1/chat/completions'), 'http://127.0.0.1:9090');
  });

  await t.test('resolveHomePath expands ~ to home directory', () => {
    assert.strictEqual(resolveHomePath('~/models'), path.join(os.homedir(), 'models'));
    assert.strictEqual(resolveHomePath('/tmp/models'), '/tmp/models');
  });

  await t.test('loadModelsConfig parses valid models.yml', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-models-test-'));
    const yamlPath = path.join(tmpDir, 'models.yml');
    const mockYaml = `
providers:
  llamacpp:
    baseUrl: http://localhost:8080/v1
    modelDir: /test/models
    llamaServer: llama-server
    models:
      - id: model-a.gguf
        name: Model-A
        reasoning: true
        contextWindow: 16384
        maxTokens: 2048
        launchArgs: ["--port", "8080", "--ctx-size", "16384"]
`;
    await fs.writeFile(yamlPath, mockYaml, 'utf8');

    const config = loadModelsConfig(yamlPath);
    assert.ok(config);
    assert.strictEqual(config.models.length, 1);
    assert.strictEqual(config.models[0].name, 'Model-A');
    assert.strictEqual(config.models[0].reasoning, true);
    assert.strictEqual(config.models[0].contextWindow, 16384);
    assert.strictEqual(config.models[0].modelPath, '/test/models/model-a.gguf');

    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  await t.test('reasoning does not count against the budget it never occupies', () => {
    const manager = new ContextManager(1000, 200); // 800 token budget
    const messages: ChatMessage[] = [
      { role: 'system', content: 'system prompt' },
      { role: 'user', content: 'hello from user 1' },
      { role: 'assistant', content: 'assistant 1', reasoning_content: 'a'.repeat(20000) },
      { role: 'user', content: 'hello from user 2' },
      { role: 'assistant', content: 'assistant 2', reasoning_content: 'b'.repeat(20000) },
    ];

    // The client never forwards reasoning_content, so a long chain of thought must not
    // evict real history to make room for itself.
    const result = manager.selectContext(messages);
    assert.strictEqual(result.trimmed, false, 'reasoning alone must not trigger compaction');
    assert.strictEqual(result.messages.length, messages.length, 'no turn was dropped');
  });

  await t.test('ContextManager trims context when real content is over budget', () => {
    const manager = new ContextManager(1000, 200); // 800 token budget
    const messages: ChatMessage[] = [
      { role: 'system', content: 'system prompt' },
      { role: 'user', content: 'hello from user 1' },
      { role: 'assistant', content: 'a'.repeat(2000) },
      { role: 'user', content: 'hello from user 2' },
      { role: 'assistant', content: 'b'.repeat(2000) },
      { role: 'user', content: 'hello from user 3' },
      { role: 'assistant', content: 'c'.repeat(2000) },
    ];

    const result = manager.selectContext(messages);
    assert.strictEqual(result.trimmed, true);
    assert.ok(result.summary);
    assert.ok(result.fits, 'compaction must bring the request under budget');
    assert.strictEqual(result.messages.at(-2)?.content, 'hello from user 3', 'the newest turn survives');
  });

  await t.test('elideToolResultContent elides long output', () => {
    const longText = Array.from({ length: 50 }, (_, i) => `line ${i + 1}`).join('\n');
    const elided = elideToolResultContent(longText, 10);
    assert.match(elided, /\[elided from context: 40 lines\]/);
    assert.match(elided, /line 1/);
    assert.match(elided, /line 50/);
  });

  await t.test('formatFooter displays AI status brackets and running timer', async () => {
    const { formatFooter } = await import('../src/tui/components.js');

    // When no model is loaded
    const unloaded = formatFooter({ cwd: '/test/dir' });
    assert.match(unloaded, /no model loaded/);
    assert.match(unloaded, /\/model to select/);
    assert.doesNotMatch(unloaded, /Qwen/);

    // Default Idle status
    const idle = formatFooter({
      modelName: 'Qwen3.5-4B',
      interactionMode: 'agent',
      cwd: '/test/dir',
      contextWindow: 16384,
      usedTokens: 1200,
    });
    assert.match(idle, /agent/);
    assert.match(idle, /\[Idle\]/);

    // Thinking status with live running timer
    const thinking = formatFooter({
      modelName: 'Qwen3.5-4B',
      interactionMode: 'agent',
      aiStatus: 'thinking',
      cwd: '/test/dir',
      contextWindow: 16384,
      usedTokens: 1200,
      turnDurationMs: 3400,
      isRunning: true,
    });
    assert.match(thinking, /\[Thinking\]/);
    assert.match(thinking, /3\.4s/);
    assert.doesNotMatch(thinking, /took/);

    // Working status
    const working = formatFooter({
      modelName: 'Qwen3.5-4B',
      interactionMode: 'plan',
      aiStatus: 'working',
      cwd: '/test/dir',
      contextWindow: 16384,
      usedTokens: 1200,
      tokensPerSecond: 25.4,
      turnDurationMs: 6200,
      isRunning: true,
    });
    assert.match(working, /plan/);
    assert.match(working, /\[Working\]/);
    assert.match(working, /25\.4 tok\/s/);

    // Serving status
    const serving = formatFooter({
      modelName: 'Qwen3.5-4B',
      aiStatus: 'serving',
      cwd: '/test/dir',
    });
    assert.match(serving, /\[Serving\]/);

    // Completed turn duration shows "took"
    const finished = formatFooter({
      modelName: 'Qwen3.5-4B',
      aiStatus: 'idle',
      cwd: '/test/dir',
      turnDurationMs: 12500,
      isRunning: false,
    });
    assert.match(finished, /took 12\.5s/);
  });

  await t.test('BannerView omits model line unless loaded', async () => {
    const { BannerView } = await import('../src/tui/components.js');

    // Without loaded model
    const bannerWithoutModel = new BannerView({
      version: '0.2.0',
      cwd: '/test/dir',
    });
    const rendered1 = bannerWithoutModel.render(80).join('\n');
    assert.match(rendered1, /⚡ DSH-Lite CLI/);
    assert.doesNotMatch(rendered1, /Model:/);

    // After setting model
    bannerWithoutModel.setModel('Qwen3.5-4B', 'thinking');
    const rendered2 = bannerWithoutModel.render(80).join('\n');
    assert.match(rendered2, /Model:/);
    assert.match(rendered2, /Qwen3\.5-4B/);
  });

  await t.test('buildServerArgs adds --host 0.0.0.0 when isHost is true', async () => {
    const { buildServerArgs } = await import('../src/llm/server.js');
    const mockModel: any = {
      modelPath: '/path/to/model.gguf',
      baseUrl: 'http://localhost:8080/v1',
      launchArgs: ['--port', '8080', '--ctx-size', '4096'],
    };

    const regularArgs = buildServerArgs(mockModel, false);
    assert.strictEqual(regularArgs.includes('--host'), false);

    const hostArgs = buildServerArgs(mockModel, true);
    assert.strictEqual(hostArgs.includes('--host'), true);
    assert.strictEqual(hostArgs[hostArgs.indexOf('--host') + 1], '0.0.0.0');
  });

  await t.test('ServeView displays remote URL and live timing logs', async () => {
    const { ServeView } = await import('../src/tui/components.js');
    const serveView = new ServeView({
      modelName: 'Ornith-1.5-9B',
      port: '8080',
      localUrl: 'http://localhost:8080/v1',
      remoteUrl: 'http://192.168.1.50:8080/v1',
    });

    serveView.addLogLine('149.46.088.705 I slot print_timing: id 0 | task 56606 | n_gen = 100, tg = 13.01 t/s');
    const rendered = serveView.render(120).join('\n');

    assert.match(rendered, /Ornith-1\.5-9B/);
    assert.match(rendered, /http:\/\/192\.168\.1\.50:8080\/v1/);
    assert.match(rendered, /print_timing/);
    assert.match(rendered, /13\.01 t\/s/);
  });

  await t.test('parseCommand handles disconnect and stop aliases', async () => {
    const { parseCommand } = await import('../src/tui/commands.js');
    assert.deepStrictEqual(parseCommand('/disconnect'), { name: 'disconnect', args: '' });
    assert.deepStrictEqual(parseCommand('/stop'), { name: 'disconnect', args: '' });
  });
});

test('Server reuse keys on the full launch configuration', async (t) => {
  const base = {
    id: 'shared.gguf',
    name: 'Shared',
    modelPath: '/models/shared.gguf',
    baseUrl: 'http://localhost:8080/v1',
    llamaServer: 'llama-server',
    reasoning: true,
    contextWindow: 32768,
    maxTokens: 4096,
    launchArgs: ['--port', '8080', '--ctx-size', '32768'],
  };

  await t.test('same weights and same args reuse the running server', () => {
    const other = { ...base, name: 'Shared-Low', maxTokens: 2048 };
    assert.strictEqual(sameLaunch(base, other), true);
  });

  await t.test('same weights but a different --ctx-size forces a restart', () => {
    const smaller = { ...base, name: 'Shared-Small', launchArgs: ['--port', '8080', '--ctx-size', '16384'] };
    assert.strictEqual(sameLaunch(base, smaller), false);
  });

  await t.test('same weights but an added --mmproj forces a restart', () => {
    const vision = { ...base, name: 'Shared-Vision', launchArgs: [...base.launchArgs, '--mmproj', '/models/proj.gguf'] };
    assert.strictEqual(sameLaunch(base, vision), false);
  });

  await t.test('different weights force a restart', () => {
    assert.strictEqual(sameLaunch(base, { ...base, modelPath: '/models/other.gguf' }), false);
  });
});

test('Config clamps a contextWindow the server cannot honour', async (t) => {
  await t.test('effectiveServerContext reads --ctx-size and divides by --parallel', () => {
    assert.strictEqual(effectiveServerContext(['--ctx-size', '32768']), 32768);
    assert.strictEqual(effectiveServerContext(['-c', '8192']), 8192);
    assert.strictEqual(effectiveServerContext(['--ctx-size', '32768', '--parallel', '2']), 16384);
    assert.strictEqual(effectiveServerContext(['--port', '8080']), undefined);
  });

  await t.test('an over-declared contextWindow is lowered and reported', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-ctx-clamp-'));
    const yamlPath = path.join(tmpDir, 'models.yml');
    await fs.writeFile(yamlPath, `
providers:
  llamacpp:
    modelDir: /test/models
    models:
      - id: over.gguf
        name: Over-Declared
        contextWindow: 131072
        maxTokens: 4096
        launchArgs: ["--ctx-size", "81980"]
      - id: fine.gguf
        name: Honest
        contextWindow: 16384
        maxTokens: 2048
        launchArgs: ["--ctx-size", "16384"]
`, 'utf8');

    const config = loadModelsConfig(yamlPath)!;
    assert.strictEqual(config.models[0].contextWindow, 81980);
    assert.strictEqual(config.models[1].contextWindow, 16384);
    assert.strictEqual(config.warnings.length, 1);
    assert.match(config.warnings[0], /Over-Declared/);

    await fs.rm(tmpDir, { recursive: true, force: true });
  });
});

test('ensure() restarts llama-server when launch args change', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-ensure-'));
  const port = 20000 + Math.floor(Math.random() * 20000);

  // Stand-in for llama-server: reports ready and echoes the args it was given.
  const fake = path.join(tmpDir, 'fake-llama.mjs');
  await fs.writeFile(fake, `
import http from 'node:http';
const args = process.argv.slice(2);
const ctx = args[args.indexOf('--ctx-size') + 1];
http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify(req.url === '/props'
    ? { model_path: args[args.indexOf('-m') + 1], n_ctx: Number(ctx) }
    : { status: 'ok', n_ctx: Number(ctx) }));
}).listen(${port});
`, 'utf8');

  const wrapper = path.join(tmpDir, 'llama-server');
  await fs.writeFile(wrapper, `#!/bin/sh\nexec "${process.execPath}" "${fake}" "$@"\n`, 'utf8');
  await fs.chmod(wrapper, 0o755);

  const weights = path.join(tmpDir, 'shared.gguf');
  await fs.writeFile(weights, 'weights', 'utf8');

  const model = (name: string, ctxSize: string) => ({
    id: name, name, modelPath: weights,
    baseUrl: `http://localhost:${port}/v1`,
    llamaServer: wrapper,
    reasoning: false,
    contextWindow: Number(ctxSize),
    maxTokens: 1024,
    launchArgs: ['--ctx-size', ctxSize],
  });

  const manager = new LlamaServerManager({
    logFile: path.join(tmpDir, 'llama.log'),
    readyTimeoutMs: 15_000,
    pollIntervalMs: 50,
  });
  const servedContext = async () =>
    (await fetch(`http://localhost:${port}/health`).then((r) => r.json())).n_ctx;

  try {
    const small = model('Shared-Vision', '16384');
    await manager.ensure(small);
    assert.strictEqual(await servedContext(), 16384);

    await t.test('an entry with identical args keeps the running server', async () => {
      const pid = (manager as any).child.pid;
      await manager.ensure({ ...small, name: 'Shared-Vision-Low', maxTokens: 512 });
      assert.strictEqual((manager as any).child.pid, pid, 'server was restarted unnecessarily');
    });

    await t.test('an entry sharing the GGUF but wanting more context restarts it', async () => {
      await manager.ensure(model('Shared-Text', '32768'));
      assert.strictEqual(await servedContext(), 32768, 'client would have budgeted against a 16384 server');
    });
  } finally {
    await manager.stop();
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
});

test('Concurrent lifecycle calls do not tear down each other server', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-race-'));
  const port = 20000 + Math.floor(Math.random() * 20000);

  // Stand-in for llama-server, slow to become ready like a real model load.
  const fake = path.join(tmpDir, 'fake-llama.mjs');
  await fs.writeFile(fake, `
import http from 'node:http';
import fs from 'node:fs';
const args = process.argv.slice(2);
fs.appendFileSync(${JSON.stringify(path.join(tmpDir, 'spawns.log'))}, process.pid + '\\n');
let ready = false;
setTimeout(() => { ready = true; }, 600);
http.createServer((req, res) => {
  if (!ready) { res.writeHead(503); res.end('{}'); return; }
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ status: 'ok', model_path: args[args.indexOf('-m') + 1] }));
}).listen(${port});
`, 'utf8');

  const wrapper = path.join(tmpDir, 'llama-server');
  await fs.writeFile(wrapper, `#!/bin/sh\nexec "${process.execPath}" "${fake}" "$@"\n`, 'utf8');
  await fs.chmod(wrapper, 0o755);

  const weights = path.join(tmpDir, 'model.gguf');
  await fs.writeFile(weights, 'weights', 'utf8');

  const model: any = {
    id: 'Racer', name: 'Racer', modelPath: weights,
    baseUrl: `http://localhost:${port}/v1`,
    llamaServer: wrapper,
    reasoning: false, contextWindow: 8192, maxTokens: 1024,
    launchArgs: ['--ctx-size', '8192'],
  };

  const manager = new LlamaServerManager({
    logFile: path.join(tmpDir, 'llama.log'),
    readyTimeoutMs: 15_000,
    pollIntervalMs: 50,
  });

  try {
    // /model starts a load; the user submits a prompt before it finishes, so
    // runPrompt calls ensure() again while the first is still in flight.
    const fromSwitch = manager.ensure(model);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const fromPrompt = manager.ensure(model);

    await assert.doesNotReject(Promise.all([fromSwitch, fromPrompt]));

    const spawns = (await fs.readFile(path.join(tmpDir, 'spawns.log'), 'utf8')).trim().split('\n');
    assert.strictEqual(spawns.length, 1, 'the second call reused the server instead of respawning');
    assert.strictEqual(manager.isRunning, true, 'a server is still running once both calls settle');
    assert.strictEqual(manager.model?.name, 'Racer');

    const health = await fetch(`http://localhost:${port}/health`);
    assert.strictEqual(health.status, 200, 'the selected model is actually reachable');
  } finally {
    await manager.stop();
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
});

test('waitUntilReady reports a server killed by a signal', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-signal-'));
  const port = 20000 + Math.floor(Math.random() * 20000);

  // Never binds a port and never exits on its own: only the kill ends it,
  // standing in for an llama-server the OS kills while loading weights.
  const fake = path.join(tmpDir, 'fake-llama.mjs');
  await fs.writeFile(fake, `setInterval(() => {}, 1000);\n`, 'utf8');

  const wrapper = path.join(tmpDir, 'llama-server');
  await fs.writeFile(wrapper, `#!/bin/sh\nexec "${process.execPath}" "${fake}" "$@"\n`, 'utf8');
  await fs.chmod(wrapper, 0o755);

  const weights = path.join(tmpDir, 'model.gguf');
  await fs.writeFile(weights, 'weights', 'utf8');

  const manager = new LlamaServerManager({
    logFile: path.join(tmpDir, 'llama.log'),
    readyTimeoutMs: 10_000,
    pollIntervalMs: 50,
  });

  const model: any = {
    id: 'Doomed', name: 'Doomed', modelPath: weights,
    baseUrl: `http://localhost:${port}/v1`,
    llamaServer: wrapper,
    reasoning: false, contextWindow: 8192, maxTokens: 1024,
    launchArgs: ['--ctx-size', '8192'],
  };

  const started = Date.now();
  const pending = manager.ensure(model);

  // Kill it the way an OOM kill would, once it is being polled for readiness.
  setTimeout(() => {
    const child = (manager as any).child;
    if (child) process.kill(child.pid, 'SIGKILL');
  }, 250);

  await assert.rejects(pending, /exited unexpectedly with signal SIGKILL/);
  assert.ok(Date.now() - started < 5000, 'failed fast instead of polling to the ready timeout');

  await manager.stop();
  await fs.rm(tmpDir, { recursive: true, force: true });
});

/** Every tool message must follow the assistant message that requested it. */
function assertWellFormed(messages: ChatMessage[], label: string) {
  const announced = new Set<string>();
  for (const message of messages) {
    for (const call of message.tool_calls ?? []) announced.add(call.id);
    if (message.role === 'tool') {
      assert.ok(
        message.tool_call_id && announced.has(message.tool_call_id),
        `${label}: tool result ${message.tool_call_id} has no assistant message requesting it`
      );
    }
  }
}

test('Compaction always produces a request that fits and is well formed', async (t) => {
  const tokens = (messages: ChatMessage[]) =>
    messages.reduce((acc, m) => acc + estimateMessageTokens(m), 0);

  await t.test('a single oversized tool result is truncated, not left over budget', () => {
    const manager = new ContextManager(22480, 12288); // Qwen3.8-27B-IQ3_XXS-xhigh
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are a coding assistant.' },
      { role: 'user', content: 'read the big file' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 't1', type: 'function', function: { name: 'view_file', arguments: '{}' } }],
      },
      { role: 'tool', tool_call_id: 't1', name: 'view_file', content: 'x'.repeat(200_000) },
    ];

    const result = manager.selectContext(messages);

    assert.strictEqual(result.fits, true);
    assert.ok(tokens(result.messages) <= manager.budgetTokens,
      `sent ${tokens(result.messages)} tokens against a ${manager.budgetTokens} budget`);
    assertWellFormed(result.messages, 'truncated');
    assert.deepStrictEqual(result.messages.map((m) => m.role), ['system', 'user', 'assistant', 'tool'],
      'the question and its tool call survive');
    assert.match(result.summary!, /truncated/);
  });

  await t.test('old turns are dropped whole, never splitting a tool call from its result', () => {
    const manager = new ContextManager(4000, 1000);
    const messages: ChatMessage[] = [{ role: 'system', content: 'sys' }];
    for (let turn = 1; turn <= 4; turn++) {
      messages.push(
        { role: 'user', content: `question ${turn}` },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: `t${turn}`, type: 'function', function: { name: 'grep_search', arguments: '{}' } }],
        },
        { role: 'tool', tool_call_id: `t${turn}`, name: 'grep_search', content: 'match\n'.repeat(900) }
      );
    }

    const result = manager.selectContext(messages);

    assert.strictEqual(result.fits, true);
    assert.ok(tokens(result.messages) <= manager.budgetTokens);
    assertWellFormed(result.messages, 'dropped turns');
    assert.strictEqual(result.messages.at(-3)?.content, 'question 4', 'the newest turn is kept');
    assert.match(result.summary!, /old turns dropped/);
  });

  await t.test('a tool result carried in from a resumed log without its assistant is dropped', () => {
    const manager = new ContextManager(2000, 500);
    const messages: ChatMessage[] = [
      { role: 'system', content: 'sys' },
      { role: 'tool', tool_call_id: 'orphan', name: 'bash', content: 'output\n'.repeat(500) },
      { role: 'user', content: 'what now?' },
    ];

    const result = manager.selectContext(messages);

    assertWellFormed(result.messages, 'resumed orphan');
    assert.strictEqual(result.messages.some((m) => m.tool_call_id === 'orphan'), false);
  });

  await t.test('the reserve for tool schemas comes out of the same budget', () => {
    const manager = new ContextManager(8000, 1000);
    const messages: ChatMessage[] = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'a'.repeat(24_000) },
    ];

    const withoutReserve = manager.selectContext(messages);
    const withReserve = manager.selectContext(messages, 3000);

    assert.ok(tokens(withReserve.messages) < tokens(withoutReserve.messages),
      'a reserved allowance leaves less room for messages');
    assert.ok(tokens(withReserve.messages) <= manager.budgetTokens - 3000);
  });

  await t.test('fits is false when the window cannot hold the turn at all', () => {
    const manager = new ContextManager(1200, 100);
    const messages: ChatMessage[] = [
      { role: 'system', content: 'instructions '.repeat(500) },
      { role: 'user', content: 'hello' },
    ];

    const result = manager.selectContext(messages);
    assert.strictEqual(result.fits, false, 'an unfittable turn is reported, not sent anyway');
  });
});

test('Model discovery falls back to the catalogue shipped with this install', () => {
  const found = findModelsConfigPath();
  assert.ok(found && fsSync.existsSync(found), 'dsh must find a models.yml from any directory');
  assert.strictEqual(/pi-lite|\/Documents\/pi\//.test(found), false,
    `discovery resolved to an unrelated project's catalogue: ${found}`);
});

test('Sampling follows the active model and mode', async (t) => {
  const captureBody = (client: DeepSeekClient) => {
    const sent: any[] = [];
    (client as any).openai = {
      chat: { completions: { create: async (body: any) => { sent.push(body); return (async function* () {})(); } } },
    };
    return sent;
  };

  // The checked-in example, not discovery: a personal models.yml is edited as models come and go,
  // and a test that keys off it fails for reasons that have nothing to do with sampling.
  const cfg = loadModelsConfig(path.join(process.cwd(), 'models.example.yml'));
  const named = (name: string) => cfg?.models.find((m) => m.name === name);

  await t.test('switching to a model without a sampling block clears the previous one', () => {
    const withSampling = named('Ministral-3-8B-Instruct');
    const withoutSampling = named('Qwen3-4B-Instruct');
    assert.ok(withSampling?.sampling && !withoutSampling?.sampling, 'fixture models changed');

    const agent = new Agent({ client: new DeepSeekClient() });
    agent.setModel(withSampling);
    assert.strictEqual(agent.client.resolveSampling().temperature, 0.05);

    agent.setModel(withoutSampling);
    assert.strictEqual(agent.client.resolveSampling().temperature, DEFAULT_SAMPLING.instruct.temperature,
      "the previous model's temperature leaked into the new one");
  });

  await t.test('/mode picks the matching half of the model sampling block', () => {
    const client = new DeepSeekClient();
    client.configureModel({
      model: 'dual',
      mode: 'thinking',
      sampling: { thinking: { temperature: 0.9 }, instruct: { temperature: 0.1 } },
      extendedSampling: true,
    });
    assert.strictEqual(client.resolveSampling().temperature, 0.9);
    client.setMode('instruct');
    assert.strictEqual(client.resolveSampling().temperature, 0.1, 'mode switch kept the old half');
    client.setMode('thinking');
    assert.strictEqual(client.resolveSampling().temperature, 0.9);
  });

  await t.test('top_k and min_p reach a llama.cpp request', async () => {
    const client = new DeepSeekClient();
    const sent = captureBody(client);
    client.configureModel({
      model: 'local',
      mode: 'instruct',
      sampling: { instruct: { temperature: 0.05, top_p: 1, top_k: 0, min_p: 0 } },
      extendedSampling: true,
    });
    await client.streamChat([{ role: 'user', content: 'hi' }]);

    assert.strictEqual(sent[0].top_k, 0);
    assert.strictEqual(sent[0].min_p, 0);
    assert.strictEqual(sent[0].temperature, 0.05);
  });

  await t.test('the cloud API is not sent llama.cpp extensions', async () => {
    const client = new DeepSeekClient();
    const sent = captureBody(client);
    client.configureModel({ model: 'deepseek-chat' });
    await client.streamChat([{ role: 'user', content: 'hi' }]);

    for (const key of ['top_k', 'min_p', 'chat_template_kwargs', 'reasoning_effort']) {
      assert.strictEqual(key in sent[0], false, `${key} would be rejected by the DeepSeek API`);
    }
    assert.strictEqual(typeof sent[0].temperature, 'number', 'standard sampling still applies');
  });

  await t.test('/mode instruct actually turns thinking off on a reasoning model', async () => {
    const client = new DeepSeekClient();
    const sent = captureBody(client);
    client.configureModel({ model: 'reasoner', mode: 'thinking', reasoning: true, extendedSampling: true });

    await client.streamChat([{ role: 'user', content: 'hi' }]);
    assert.strictEqual(sent[0].chat_template_kwargs.enable_thinking, true);

    client.setMode('instruct');
    await client.streamChat([{ role: 'user', content: 'hi' }]);
    assert.strictEqual(sent[0 + 1].chat_template_kwargs.enable_thinking, false);
  });

  await t.test('an instruct-only model gets no thinking switch', async () => {
    const client = new DeepSeekClient();
    const sent = captureBody(client);
    client.configureModel({ model: 'plain', mode: 'instruct', reasoning: false, extendedSampling: true });
    await client.streamChat([{ role: 'user', content: 'hi' }]);
    assert.strictEqual('chat_template_kwargs' in sent[0], false);
  });

  await t.test("a model's reasoning_effort merges with the thinking switch", async () => {
    const xhigh = named('Qwen3.8-27B-xhigh');
    assert.ok(xhigh, 'fixture model changed');

    const agent = new Agent({ client: new DeepSeekClient() });
    agent.setModel(xhigh);
    // setModel rebuilds the HTTP client for the model endpoint, so stub after it.
    const sent = captureBody(agent.client);
    await agent.client.streamChat([{ role: 'user', content: 'hi' }]);

    assert.deepStrictEqual(sent[0].chat_template_kwargs, { enable_thinking: true, reasoning_effort: 'xhigh' });
  });
})

test('The checked-in example catalogue is valid and machine-independent', async (t) => {
  const examplePath = path.join(process.cwd(), 'models.example.yml');

  await t.test('it parses into usable models', () => {
    const config = loadModelsConfig(examplePath);
    assert.ok(config && config.models.length > 0, 'a fresh clone must find a catalogue');
    assert.deepStrictEqual(config.warnings, [], 'the example must not declare a window its args cannot honour');
    for (const model of config.models) {
      assert.ok(model.name && model.modelPath && model.launchArgs.length > 0, `${model.name} is incomplete`);
    }
  });

  await t.test('it names no particular machine', async () => {
    const raw = await fs.readFile(examplePath, 'utf8');
    assert.doesNotMatch(raw, /\/Users\/[a-z]+\//i, 'absolute home paths do not transfer between machines');
  });

  await t.test('it demonstrates every field the loader understands', () => {
    const models = loadModelsConfig(examplePath)!.models;
    assert.ok(models.some((m) => m.reasoning), 'a reasoning model');
    assert.ok(models.some((m) => !m.reasoning), 'an instruct model');
    assert.ok(models.some((m) => m.sampling?.instruct?.temperature !== undefined), 'per-mode sampling');
    assert.ok(models.some((m) => m.sampling?.thinking?.extra), 'chat_template_kwargs via extra');
    assert.ok(models.some((m) => m.launchArgs.includes('--mmproj')), 'a vision projector');
  });
});

test('A server that reports healthy after a GPU failure is refused', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-oom-'));
  const port = 20000 + Math.floor(Math.random() * 20000);

  // Reproduces the real llama-server sequence: the Metal warmup decode fails out of memory,
  // then the server finishes starting and answers /health with 200 regardless.
  const fake = path.join(tmpDir, 'fake-llama.mjs');
  await fs.writeFile(fake, `
import http from 'node:http';
console.error('ggml_metal_synchronize: error: command buffer 0 failed with status 5');
console.error('error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)');
console.error('llama_decode: failed to decode, ret = -3');
setTimeout(() => {
  console.log('srv  llama_server: model loaded');
  http.createServer((_q, r) => { r.writeHead(200, {'content-type':'application/json'}); r.end('{"status":"ok"}'); }).listen(${port});
}, 150);
`, 'utf8');

  const wrapper = path.join(tmpDir, 'llama-server');
  await fs.writeFile(wrapper, `#!/bin/sh\nexec "${process.execPath}" "${fake}" "$@"\n`, 'utf8');
  await fs.chmod(wrapper, 0o755);
  const weights = path.join(tmpDir, 'model.gguf');
  await fs.writeFile(weights, 'weights', 'utf8');

  const manager = new LlamaServerManager({
    logFile: path.join(tmpDir, 'llama.log'),
    readyTimeoutMs: 10_000,
    pollIntervalMs: 50,
  });

  const model: any = {
    id: 'TooBig', name: 'TooBig', modelPath: weights,
    baseUrl: `http://localhost:${port}/v1`,
    llamaServer: wrapper,
    reasoning: false, contextWindow: 22480, maxTokens: 8192,
    launchArgs: ['--ctx-size', '22480'],
  };

  try {
    await t.test('the failure is reported instead of a ready server', async () => {
      await assert.rejects(
        manager.ensure(model),
        /cannot serve: the GPU ran out of memory/,
        'a healthy status must not outrank the server saying it failed'
      );
    });

    await t.test('the message says what to change', async () => {
      const error = await manager.ensure(model).catch((err: Error) => err.message);
      assert.match(error as string, /--ctx-size or --ubatch-size/);
    });

    await t.test('nothing is left marked active', () => {
      assert.strictEqual(manager.model, undefined);
      assert.strictEqual(manager.isRunning, false, 'the broken server must be torn down');
    });
  } finally {
    await manager.stop();
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
});

test('Server output classification', async (t) => {
  await t.test('a fit warning alone is not fatal', () => {
    // This precedes plenty of loads that go on to work.
    const manager = new LlamaServerManager();
    void manager;
    assert.strictEqual(
      findFatalOutput('W common_fit_params: failed to fit params to free device memory: abort'),
      undefined
    );
  });

  await t.test('routine startup noise is not fatal', () => {
    for (const line of [
      'srv llama_server: CORS is set to allow all origins',
      'srv load_model: initializing, n_slots = 1',
      'W srv init: chat template supports preserving reasoning',
      'srv llama_server: listening on http://127.0.0.1:8080',
    ]) {
      assert.strictEqual(findFatalOutput(line), undefined, line);
    }
  });

  await t.test('real failures are caught', () => {
    assert.match(findFatalOutput('error: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)')!, /GPU ran out of memory/);
    assert.match(findFatalOutput('ggml_metal_graph_compute: backend is in error state')!, /cannot recover/);
    assert.match(findFatalOutput('E llama_decode: failed to decode, ret = -3')!, /forward pass/);
    assert.match(findFatalOutput('error loading model architecture')!, /could not be loaded/);
  });
});
