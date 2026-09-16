import { test } from 'node:test';
import assert from 'node:assert';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { Agent, CANCELLED_TOOL_RESULT, type AgentTurnCallbacks } from '../src/agent.js';
import { DeepSeekClient } from '../src/llm/client.js';
import type { ChatMessage } from '../src/llm/types.js';
import { estimateMessageTokens, estimateTokens } from '../src/context.js';

test('Agent turn callbacks receive tool events without stdout writes', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-agent-test-'));

  const client = new DeepSeekClient();
  // Mock streamChat to simulate a 2-step turn:
  // Step 1: LLM calls write_file
  // Step 2: LLM finishes with answer
  let callCount = 0;
  client.streamChat = async (_messages, _tools, callbacks) => {
    callCount++;
    if (callCount === 1) {
      callbacks.onReasoningDelta?.('Thinking about writing file...');
      return {
        content: '',
        reasoningContent: 'Thinking about writing file...',
        toolCalls: [
          {
            id: 'call_1',
            type: 'function',
            function: {
              name: 'write_file',
              arguments: JSON.stringify({ path: 'calc.txt', content: 'calculator' }),
            },
          },
        ],
      };
    } else {
      callbacks.onContentDelta?.('File has been written.');
      return {
        content: 'File has been written.',
        toolCalls: [],
      };
    }
  };

  const agent = new Agent({
    client,
    cwd: tmpDir,
  });

  const events: string[] = [];
  const stepMetrics: any[] = [];
  const callbacks: AgentTurnCallbacks = {
    onAssistantStart: () => events.push('assistant_start'),
    onAssistantEnd: () => events.push('assistant_end'),
    onStepComplete: (metrics) => {
      events.push('step_complete');
      stepMetrics.push(metrics);
    },
    onReasoningDelta: (delta) => events.push(`reasoning:${delta}`),
    onContentDelta: (delta) => events.push(`content:${delta}`),
    onToolStart: (_id, name, summary) => events.push(`tool_start:${name}:${summary}`),
    onToolEnd: (_id, name, exec) => events.push(`tool_end:${name}:${exec.isError}`),
  };

  await agent.runTurn('Create calc.txt', callbacks);

  assert.deepStrictEqual(events, [
    'assistant_start',
    'reasoning:Thinking about writing file...',
    'assistant_end',
    'step_complete',
    'tool_start:write_file:calc.txt',
    'tool_end:write_file:false',
    'assistant_start',
    'content:File has been written.',
    'assistant_end',
    'step_complete',
  ]);

  assert.strictEqual(stepMetrics.length, 2);
  assert.ok(stepMetrics[0].turnDurationMs >= 0);
  assert.ok(stepMetrics[0].stepDurationMs >= 0);
  assert.ok(stepMetrics[1].turnDurationMs >= stepMetrics[0].turnDurationMs);

  const createdContent = await fs.readFile(path.join(tmpDir, 'calc.txt'), 'utf8');
  assert.strictEqual(createdContent, 'calculator');
});

test('Agent runTurn respects abortSignal and invokes onCancelled', async () => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-agent-abort-'));

  const client = new DeepSeekClient();
  const controller = new AbortController();

  client.streamChat = async (_messages, _tools, _callbacks, signal) => {
    // Simulate abort during chat
    controller.abort();
    const error = new Error('Request was aborted');
    error.name = 'AbortError';
    throw error;
  };

  const agent = new Agent({
    client,
    cwd: tmpDir,
  });

  let cancelledCalled = false;
  const callbacks: AgentTurnCallbacks = {
    onCancelled: () => {
      cancelledCalled = true;
    },
  };

  await agent.runTurn('Run long task', callbacks, controller.signal);
  assert.strictEqual(cancelledCalled, true, 'Expected onCancelled to be invoked');
});

test('Agent interaction modes filter tools and configure prompt correctly', async () => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-mode-test-'));

  const client = new DeepSeekClient();
  let toolsReceived: any[] = [];
  client.streamChat = async (_messages, tools) => {
    toolsReceived = tools;
    return {
      content: 'OK',
      toolCalls: [],
    };
  };

  const agent = new Agent({ client, cwd: tmpDir });

  // 1. Agent mode (default) -> all tools
  await agent.runTurn('Do work', {});
  assert.strictEqual(agent.interactionMode, 'agent');
  assert.ok(toolsReceived.length >= 6);
  assert.ok(toolsReceived.some((t) => t.function.name === 'write_file'));
  assert.ok(toolsReceived.some((t) => t.function.name === 'bash'));

  // 2. Plan mode -> only read-only tools
  agent.setInteractionMode('plan');
  assert.strictEqual(agent.interactionMode, 'plan');
  await agent.runTurn('Plan project', {});
  const toolNames = toolsReceived.map((t) => t.function.name);
  assert.deepStrictEqual(toolNames.sort(), ['grep_search', 'list_dir', 'view_file', 'web_fetch', 'web_search']);
  assert.strictEqual(toolNames.includes('write_file'), false);
  assert.strictEqual(toolNames.includes('bash'), false);
  assert.ok(agent.messages[0].content?.includes('PLANNING ASSISTANT') || agent.messages[0].content?.includes('plan'));

  // 3. Chat mode -> the web tools only, so it can still look things up
  agent.setInteractionMode('chat');
  assert.strictEqual(agent.interactionMode, 'chat');
  await agent.runTurn('Hello', {});
  const chatToolNames = toolsReceived.map((t) => t.function.name);
  assert.deepStrictEqual(chatToolNames.sort(), ['web_fetch', 'web_search']);
  assert.strictEqual(chatToolNames.includes('view_file'), false, 'chat must not read the workspace');
  assert.strictEqual(chatToolNames.includes('write_file'), false);
  assert.strictEqual(chatToolNames.includes('bash'), false);
  assert.ok(agent.messages[0].content?.includes('conversational'));
  assert.ok(
    agent.messages[0].content?.includes('web_search'),
    'the chat prompt must tell the model it can reach the internet'
  );
});

test('Agent handles maxSteps and emits onNotice when limit reached', async () => {
  const client = new DeepSeekClient();
  let stepCount = 0;
  client.streamChat = async () => {
    stepCount++;
    return {
      content: 'running tool',
      toolCalls: [
        {
          id: `call_${stepCount}`,
          type: 'function',
          function: { name: 'list_dir', arguments: '{}' },
        },
      ],
    };
  };

  const agent = new Agent({ client, maxSteps: 3 });
  assert.strictEqual(agent.maxSteps, 3);

  const notices: string[] = [];
  await agent.runTurn('Loop test', {
    onNotice: (msg) => notices.push(msg),
  });

  assert.strictEqual(stepCount, 3);
  assert.strictEqual(notices.length, 1);
  assert.ok(notices[0].includes('Reached turn step limit (3 steps)'));
});


test('Agent answers every tool_call when a turn is aborted mid-loop', async () => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-agent-abort-tools-'));

  const client = new DeepSeekClient();
  const controller = new AbortController();
  const sent: ChatMessage[][] = [];
  let callCount = 0;

  client.streamChat = async (messages) => {
    sent.push(messages.map((m) => ({ ...m })));
    if (++callCount === 1) {
      return {
        content: '',
        reasoningContent: '',
        toolCalls: [
          { id: 'call_1', type: 'function', function: { name: 'list_dir', arguments: '{}' } },
          { id: 'call_2', type: 'function', function: { name: 'list_dir', arguments: '{}' } },
        ],
      };
    }
    return { content: 'Four.', reasoningContent: '', toolCalls: [] };
  };

  const agent = new Agent({ client, cwd: tmpDir });

  // Abort once the first tool has run, so its side effects are already done.
  const execute = agent.registry.execute.bind(agent.registry);
  let executed = 0;
  agent.registry.execute = async (...args) => {
    const result = await execute(...args);
    if (++executed === 1) controller.abort();
    return result;
  };

  let cancelled = 0;
  await agent.runTurn('Do two things', { onCancelled: () => cancelled++ }, controller.signal);

  assert.strictEqual(cancelled, 1, 'cancellation is reported once, not once per pending tool call');
  assert.strictEqual(executed, 1, 'no further tool runs after the abort');

  const answered = agent.messages.filter((m) => m.role === 'tool').map((m) => m.tool_call_id);
  assert.deepStrictEqual(answered, ['call_1', 'call_2'], 'both tool_call ids are answered');
  assert.notStrictEqual(agent.messages.at(-2)?.content, CANCELLED_TOOL_RESULT, 'the tool that ran keeps its real result');
  assert.strictEqual(agent.messages.at(-1)?.content, CANCELLED_TOOL_RESULT);

  // The next turn must send a well-formed conversation.
  await agent.runTurn('Never mind, what is 2+2?', {});

  const lastRequest = sent.at(-1)!;
  const pending = new Set<string>();
  for (const message of lastRequest) {
    for (const call of message.tool_calls ?? []) pending.add(call.id);
    if (message.role === 'tool' && message.tool_call_id) pending.delete(message.tool_call_id);
  }
  assert.deepStrictEqual([...pending], [], 'no tool_call is left dangling in the request');
});

test('Agent refuses to send a turn that cannot fit the context window', async () => {
  const client = new DeepSeekClient();
  let requests = 0;
  client.streamChat = async () => {
    requests++;
    return { content: 'should never run', reasoningContent: '', toolCalls: [] };
  };

  const agent = new Agent({ client, contextWindow: 1200, maxTokens: 100 });
  agent.sessionStore.appendMessage = async () => {};
  // A system prompt that alone outgrows the window leaves nothing to compact.
  agent.messages[0].content = 'instructions '.repeat(500);

  await assert.rejects(
    agent.runTurn('hello', {}),
    /does not fit .*1200-token context window/,
    'an unfittable turn is reported instead of sent'
  );
  assert.strictEqual(requests, 0, 'no request reached the model');
});

test('Agent charges the tool schemas against the context budget', async () => {
  const client = new DeepSeekClient();
  const promptTokens: number[] = [];
  client.streamChat = async (messages) => {
    promptTokens.push(messages.reduce((acc, m) => acc + estimateMessageTokens(m), 0));
    return { content: 'ok', reasoningContent: '', toolCalls: [] };
  };

  const agent = new Agent({ client, contextWindow: 16000, maxTokens: 2000 });
  agent.sessionStore.appendMessage = async () => {};
  agent.messages.push({ role: 'user', content: 'x'.repeat(60_000) });

  await agent.runTurn('and now?', {});

  const schemaTokens = estimateTokens(JSON.stringify(agent.registry.getOpenAITools()));
  assert.ok(schemaTokens > 0);
  assert.ok(
    promptTokens[0] <= agent.contextManager.budgetTokens - schemaTokens,
    `messages used ${promptTokens[0]} tokens, leaving no room for ${schemaTokens} tokens of tool schemas`
  );
});

test('Resuming a session restores history under a fresh system prompt', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-resume-'));
  const client = new DeepSeekClient();
  const sent: ChatMessage[][] = [];
  client.streamChat = async (messages) => {
    sent.push(messages.map((m) => ({ ...m })));
    return { content: 'ok', reasoningContent: '', toolCalls: [] };
  };

  // Record a session the way runTurn does: user, assistant and tool only.
  const recorder = new Agent({ client, cwd: tmpDir });
  const sessionId = recorder.sessionId;
  await recorder.sessionStore.appendMessage(sessionId, { role: 'user', content: 'first question' });
  await recorder.sessionStore.appendMessage(sessionId, { role: 'assistant', content: 'first answer' });

  await t.test('the stored log carries no system message', async () => {
    const stored = await recorder.sessionStore.loadSession(sessionId);
    assert.deepStrictEqual(stored.map((m) => m.role), ['user', 'assistant']);
  });

  await t.test('resume prepends the current system prompt and reports the count', async () => {
    const agent = new Agent({ client, cwd: tmpDir });
    const restored = await agent.resume(sessionId);

    assert.strictEqual(restored, 2);
    assert.strictEqual(agent.sessionId, sessionId);
    assert.deepStrictEqual(agent.messages.map((m) => m.role), ['system', 'user', 'assistant']);
    assert.match(agent.messages[0].content!, /coding assistant/);
  });

  await t.test('the resumed system prompt describes the session running now', async () => {
    const agent = new Agent({ client, cwd: tmpDir });
    agent.setInteractionMode('plan');
    await agent.resume(sessionId);

    assert.strictEqual(agent.messages[0].role, 'system');
    assert.match(agent.messages[0].content!, /planning assistant/i, 'the current mode wins over the recorded one');
  });

  await t.test('the next request after a resume carries the system prompt', async () => {
    const agent = new Agent({ client, cwd: tmpDir });
    await agent.resume(sessionId);
    await agent.runTurn('follow-up', {});

    const request = sent.at(-1)!;
    assert.strictEqual(request[0].role, 'system');
    assert.deepStrictEqual(request.map((m) => m.role), ['system', 'user', 'assistant', 'user']);
  });

  await t.test('a log whose own system message is stale is replaced, not duplicated', async () => {
    const staleId = recorder.sessionStore.createSessionId();
    await recorder.sessionStore.appendMessage(staleId, { role: 'system', content: 'OUTDATED PROMPT' });
    await recorder.sessionStore.appendMessage(staleId, { role: 'user', content: 'hi' });

    const agent = new Agent({ client, cwd: tmpDir });
    await agent.resume(staleId);

    assert.strictEqual(agent.messages.filter((m) => m.role === 'system').length, 1);
    assert.strictEqual(agent.messages[0].content!.includes('OUTDATED PROMPT'), false);
  });

  await t.test('resuming an empty log still switches the session being appended to', async () => {
    const emptyId = recorder.sessionStore.createSessionId();
    await fs.writeFile(path.join(tmpDir, '.dsh', 'sessions', `${emptyId}.jsonl`), '', 'utf8');

    const agent = new Agent({ client, cwd: tmpDir });
    const restored = await agent.resume(emptyId);

    assert.strictEqual(restored, 0);
    assert.strictEqual(agent.sessionId, emptyId, 'later messages must land in the session the user asked for');
    assert.deepStrictEqual(agent.messages.map((m) => m.role), ['system']);
  });

  await t.test('a session aborted mid-tool-call resumes into a well-formed request', async () => {
    const abortedId = recorder.sessionStore.createSessionId();
    const store = recorder.sessionStore;
    await store.appendMessage(abortedId, { role: 'user', content: 'do it' });
    await store.appendMessage(abortedId, {
      role: 'assistant',
      content: null,
      tool_calls: [{ id: 'c1', type: 'function', function: { name: 'bash', arguments: '{}' } }],
    });
    // The tool result never landed, so this log opens on an unanswered call.
    await store.appendMessage(abortedId, { role: 'tool', tool_call_id: 'ghost', name: 'bash', content: 'stray' });

    const agent = new Agent({ client, cwd: tmpDir });
    await agent.resume(abortedId);
    await agent.runTurn('never mind', {});

    const request = sent.at(-1)!;
    const announced = new Set<string>();
    for (const message of request) {
      for (const call of message.tool_calls ?? []) announced.add(call.id);
      if (message.role === 'tool') {
        assert.ok(announced.has(message.tool_call_id!), `stray tool result ${message.tool_call_id} was sent`);
      }
    }
  });

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('A stalled stream ends with a diagnosis instead of hanging', async () => {
  const client = new DeepSeekClient({ streamIdleTimeoutMs: 300 });
  // A server that accepts the request, sends one chunk, then goes quiet forever.
  (client as unknown as { openai: unknown }).openai = {
    chat: {
      completions: {
        create: async (_body: unknown, options: { signal?: AbortSignal }) => ({
          async *[Symbol.asyncIterator]() {
            yield { choices: [{ delta: { content: 'thinking' } }] };
            await new Promise((_resolve, reject) => {
              options.signal?.addEventListener('abort', () => reject(options.signal!.reason), { once: true });
            });
          },
        }),
      },
    },
  };
  client.configureModel({ model: 'stalled-model' });

  const started = Date.now();
  await assert.rejects(
    client.streamChat([{ role: 'user', content: 'hi' }]),
    /stopped responding: no output for 0s|stopped responding/,
    'a silent server must not hold the turn open'
  );
  assert.ok(Date.now() - started < 5000, 'gave up promptly rather than waiting out the request timeout');
});

test('An empty model response is reported rather than shown as nothing', async () => {
  const client = new DeepSeekClient();
  client.streamChat = async () => ({ content: '', reasoningContent: '', toolCalls: [] });

  const agent = new Agent({ client });
  agent.sessionStore.appendMessage = async () => {};

  const notices: string[] = [];
  await agent.runTurn('hello?', { onNotice: (message) => notices.push(message) });

  assert.strictEqual(notices.length, 1, 'an empty turn must say something');
  assert.match(notices[0], /empty response/);
});

test('A normal response produces no empty-response notice', async () => {
  const client = new DeepSeekClient();
  client.streamChat = async () => ({ content: 'here you go', reasoningContent: '', toolCalls: [] });

  const agent = new Agent({ client });
  agent.sessionStore.appendMessage = async () => {};

  const notices: string[] = [];
  await agent.runTurn('hello?', { onNotice: (message) => notices.push(message) });
  assert.deepStrictEqual(notices, []);
});

test('/clear starts an empty conversation, not just an empty screen', async (t) => {
  const client = new DeepSeekClient();
  const sent: ChatMessage[][] = [];
  client.streamChat = async (messages) => {
    sent.push(messages.map((m) => ({ ...m })));
    return { content: 'x'.repeat(8000), reasoningContent: '', toolCalls: [] };
  };

  const agent = new Agent({ client });
  agent.sessionStore.appendMessage = async () => {};
  for (let turn = 0; turn < 3; turn++) await agent.runTurn(`question ${turn}`, {});

  const before = agent.messages.length;
  const sessionBefore = agent.sessionId;
  assert.ok(before > 5, 'the conversation should have grown');

  agent.clearHistory();

  await t.test('the conversation is actually dropped', () => {
    assert.deepStrictEqual(agent.messages.map((m) => m.role), ['system']);
  });

  await t.test('the next request carries no trace of the old turns', async () => {
    await agent.runTurn('fresh question', {});
    const request = sent.at(-1)!;
    assert.deepStrictEqual(request.map((m) => m.role), ['system', 'user']);
    assert.strictEqual(request[1].content, 'fresh question');
    assert.strictEqual(
      request.some((m) => typeof m.content === 'string' && m.content.includes('question 0')),
      false,
      'an old turn would be re-sent and re-processed by the server'
    );
  });

  await t.test('a new session receives the messages, and the footer is reset', () => {
    assert.notStrictEqual(agent.sessionId, sessionBefore);
    // lastTurnMetrics is set again by the turn above; what matters is that clearing wiped it.
    const other = new Agent({ client });
    other.lastTurnMetrics = { promptTokens: 1, completionTokens: 1, totalTokens: 999, turnDurationMs: 1 };
    other.clearHistory();
    assert.strictEqual(other.lastTurnMetrics, undefined, 'the footer would keep showing a stale count');
  });

  await t.test('a mode switch still keeps the conversation', async () => {
    const keeper = new Agent({ client });
    keeper.sessionStore.appendMessage = async () => {};
    await keeper.runTurn('remember this', {});
    const lengthBefore = keeper.messages.length;
    keeper.setInteractionMode('plan');
    assert.strictEqual(keeper.messages.length, lengthBefore, 'switching mode must not discard history');
    assert.strictEqual(keeper.messages[0].role, 'system');
    assert.match(keeper.messages[0].content!, /planning assistant/i);
  });
});
