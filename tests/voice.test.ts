import { test } from 'node:test';
import assert from 'node:assert';
import * as fs from 'node:fs/promises';
import * as fsSync from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import { parseVoiceCommand, summariseForSpeech, VoiceSocket } from '../src/voice/socket.js';
import { Agent } from '../src/agent.js';
import { DeepSeekClient } from '../src/llm/client.js';
import { writeFileTool } from '../src/tools/fs.js';

test('The voice protocol accepts what a daemon sends and rejects the rest', async (t) => {
  await t.test('the three commands parse', () => {
    assert.deepStrictEqual(parseVoiceCommand('{"type":"text","text":"save it as notes.md"}'), {
      type: 'text',
      text: 'save it as notes.md',
    });
    assert.deepStrictEqual(parseVoiceCommand('{"type":"submit"}'), { type: 'submit' });
    assert.deepStrictEqual(parseVoiceCommand('{"type":"abort"}'), { type: 'abort' });
  });

  await t.test('malformed input is described, not thrown', () => {
    assert.match(parseVoiceCommand('not json') as string, /not valid JSON/);
    assert.match(parseVoiceCommand('"a string"') as string, /expected a JSON object/);
    assert.match(parseVoiceCommand('{"type":"text"}') as string, /need a "text" string/);
    assert.match(parseVoiceCommand('{"type":"rm -rf"}') as string, /unknown type/);
  });
});

test('Replies are condensed to something worth speaking', async (t) => {
  await t.test('a long answer is cut to two sentences', () => {
    const reply = 'I saved the template. It has four sections. You can edit it later. Ask if you need more.';
    assert.strictEqual(summariseForSpeech(reply), 'I saved the template. It has four sections.');
  });

  await t.test('code blocks are not read aloud', () => {
    const summary = summariseForSpeech('Done.\n```ts\nconst x = 1;\n```\nSaved as a.ts.');
    assert.doesNotMatch(summary, /const x/);
  });

  await t.test('an empty reply still says something', () => {
    assert.strictEqual(summariseForSpeech(''), 'Done.');
  });

  await t.test('a single long sentence is capped', () => {
    const summary = summariseForSpeech(`I ${'really '.repeat(200)}did it.`);
    assert.ok(summary.length <= 320);
  });
});

test('The voice socket drives the harness like a keyboard', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-voice-'));
  const socketPath = path.join(tmpDir, 'input.sock');
  const socket = new VoiceSocket(socketPath);

  const received: string[] = [];
  await socket.open({
    onText: (text) => received.push(`text:${text}`),
    onSubmit: () => received.push('submit'),
    onAbort: () => received.push('abort'),
  });

  const client = net.connect(socketPath);
  const replies: string[] = [];
  client.setEncoding('utf8');
  client.on('data', (chunk: string) => replies.push(...chunk.trim().split('\n')));
  await new Promise((resolve) => client.once('connect', resolve));
  const settle = () => new Promise((resolve) => setTimeout(resolve, 60));

  await t.test('it is owner-only, not reachable from off the machine', () => {
    const mode = fsSync.statSync(socketPath).mode & 0o777;
    assert.strictEqual(mode, 0o600, `socket mode was ${mode.toString(8)}`);
  });

  await t.test('commands arrive in order, including several in one packet', async () => {
    client.write('{"type":"text","text":"write a template"}\n{"type":"submit"}\n');
    client.write('{"type":"abort"}\n');
    await settle();
    assert.deepStrictEqual(received, ['text:write a template', 'submit', 'abort']);
  });

  await t.test('a command split across packets still arrives once', async () => {
    received.length = 0;
    client.write('{"type":"text","te');
    await settle();
    assert.deepStrictEqual(received, [], 'a partial line must not dispatch');
    client.write('xt":"and save it"}\n');
    await settle();
    assert.deepStrictEqual(received, ['text:and save it']);
  });

  await t.test('a bad line is answered and the connection survives', async () => {
    received.length = 0;
    replies.length = 0;
    client.write('{"type":"nonsense"}\n{"type":"submit"}\n');
    await settle();
    assert.match(replies.join(' '), /"type":"error"/);
    assert.deepStrictEqual(received, ['submit'], 'the good line after it still ran');
  });

  await t.test('events reach the daemon to speak', async () => {
    replies.length = 0;
    socket.broadcast({ type: 'ack' });
    socket.broadcast({ type: 'done', summary: 'All done, the file is saved.' });
    await settle();
    assert.deepStrictEqual(replies.map((r) => JSON.parse(r)), [
      { type: 'ack' },
      { type: 'done', summary: 'All done, the file is saved.' },
    ]);
  });

  await t.test('a second harness cannot steal a live socket', async () => {
    const rival = new VoiceSocket(socketPath);
    await assert.rejects(
      rival.open({ onText: () => {}, onSubmit: () => {}, onAbort: () => {} }),
      /already in use by another dsh/
    );
  });

  await t.test('closing removes the socket file', async () => {
    await socket.close();
    assert.strictEqual(fsSync.existsSync(socketPath), false);
  });

  await t.test('a socket left by a crashed harness is reclaimed', async () => {
    await fs.writeFile(socketPath, '');
    const revived = new VoiceSocket(socketPath);
    await revived.open({ onText: () => {}, onSubmit: () => {}, onAbort: () => {} });
    assert.strictEqual(revived.isOpen, true);
    await revived.close();
  });

  client.destroy();
  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('Voice mode cannot destroy anything', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-voice-tools-'));
  const client = new DeepSeekClient();
  let offered: string[] = [];
  client.streamChat = async (_messages, tools) => {
    offered = tools.map((t) => t.function.name);
    return { content: 'Saved it as notes.md.', reasoningContent: '', toolCalls: [] };
  };

  const agent = new Agent({ client, cwd: tmpDir });
  agent.sessionStore.appendMessage = async () => {};
  agent.setInteractionMode('voice');
  await agent.runTurn('write me a template and save it as notes.md', {});

  await t.test('no shell', () => {
    assert.strictEqual(offered.includes('bash'), false, 'a shell cannot be made non-destructive by filtering');
  });

  await t.test('it can still read, search, write and edit', () => {
    assert.deepStrictEqual(offered.sort(), [
      'edit_file', 'grep_search', 'list_dir', 'view_file', 'web_fetch', 'web_search', 'write_file',
    ]);
  });

  await t.test('write_file will not replace an existing file', async () => {
    const existing = path.join(tmpDir, 'keep.md');
    await fs.writeFile(existing, 'original content\n');

    const refused = await writeFileTool.execute(
      { path: 'keep.md', content: 'clobbered' },
      { cwd: tmpDir, preventOverwrite: true }
    );
    assert.match(refused, /already exists/);
    assert.match(refused, /edit_file/, 'the model needs to be told what to do instead');
    assert.strictEqual(await fs.readFile(existing, 'utf8'), 'original content\n', 'content was lost');
  });

  await t.test('it can still create a new file', async () => {
    const created = await writeFileTool.execute(
      { path: 'fresh.md', content: '# Fresh' },
      { cwd: tmpDir, preventOverwrite: true }
    );
    assert.match(created, /Successfully wrote/);
    assert.strictEqual(await fs.readFile(path.join(tmpDir, 'fresh.md'), 'utf8'), '# Fresh');
  });

  await t.test('agent mode is unaffected', async () => {
    agent.setInteractionMode('agent');
    await agent.runTurn('run the tests', {});
    assert.ok(offered.includes('bash'));
  });

  await t.test('the prompt tells the model its replies are spoken', () => {
    agent.setInteractionMode('voice');
    assert.match(agent.messages[0].content!, /read aloud/);
    assert.match(agent.messages[0].content!, /one or two short sentences/);
  });

  await fs.rm(tmpDir, { recursive: true, force: true });
});
