import { test } from 'node:test';
import assert from 'node:assert';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { SessionStore } from '../src/session/store.js';

test('SessionStore persists and recovers conversation sessions', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-session-test-'));
  const store = new SessionStore(tmpDir);

  const sessionId = store.createSessionId();
  assert.match(sessionId, /^session-\d+-[a-f0-9]+/);

  await t.test('appends and reloads messages', async () => {
    await store.appendMessage(sessionId, {
      role: 'user',
      content: 'hello from user',
    });

    await store.appendMessage(sessionId, {
      role: 'assistant',
      content: 'hello from assistant',
      reasoning_content: 'thinking about hello',
    });

    const messages = await store.loadSession(sessionId);
    assert.strictEqual(messages.length, 2);
    assert.strictEqual(messages[0].role, 'user');
    assert.strictEqual(messages[0].content, 'hello from user');
    assert.strictEqual(messages[1].role, 'assistant');
    assert.strictEqual(messages[1].content, 'hello from assistant');
    assert.strictEqual(messages[1].reasoning_content, 'thinking about hello');
  });

  await t.test('lists sessions with a preview instead of invented metadata', async () => {
    const sessions = await store.listSessions();
    assert.ok(sessions.length >= 1);
    const found = sessions.find(s => s.id === sessionId);
    assert.ok(found);
    assert.ok(found.sizeBytes > 0);
    assert.strictEqual(found.preview, 'hello from user', 'the first user message identifies the session');
  });

  await t.test('the session directory hides itself from git', async () => {
    const ignore = await fs.readFile(path.join(tmpDir, '.dsh', '.gitignore'), 'utf8');
    assert.strictEqual(ignore.trim(), '*', 'transcripts must not show up as untracked files');
  });

  // Cleanup
  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('Listing sessions does not read every transcript in full', async () => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-session-scale-'));
  const store = new SessionStore(tmpDir);

  // One large transcript: listing must describe it without pulling it into memory.
  const id = store.createSessionId();
  await store.appendMessage(id, { role: 'user', content: 'summarise the build logs' });
  for (let i = 0; i < 400; i++) {
    await store.appendMessage(id, { role: 'assistant', content: 'x'.repeat(5000) });
  }
  const bytes = (await fs.stat(path.join(tmpDir, '.dsh', 'sessions', `${id}.jsonl`))).size;
  assert.ok(bytes > 2_000_000, 'fixture should be large');

  const [session] = await store.listSessions();
  assert.strictEqual(session.id, id);
  assert.strictEqual(session.sizeBytes, bytes, 'size comes from stat, not from reading');
  assert.strictEqual(session.preview, 'summarise the build logs');

  await fs.rm(tmpDir, { recursive: true, force: true });
});
