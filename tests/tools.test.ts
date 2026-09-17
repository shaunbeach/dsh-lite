import { test } from 'node:test';
import assert from 'node:assert';
import * as fs from 'node:fs/promises';
import * as fsSync from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { ToolRegistry } from '../src/tools/registry.js';
import { bashTool } from '../src/tools/bash.js';
import { editFileTool } from '../src/tools/edit.js';
import { viewFileTool } from '../src/tools/fs.js';
import { grepSearchTool } from '../src/tools/grep.js';
import { listDirTool } from '../src/tools/glob.js';
import { Agent, resolveMaxSteps } from '../src/agent.js';
import { toolOutputLimitBytes } from '../src/tools/limits.js';
import { judgeCommand, refusalMessage } from '../src/tools/destructive.js';
import { renderDiff } from '../src/ui/diff.js';

test('ToolRegistry executes essential coding tools', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-test-'));
  const registry = new ToolRegistry();
  const context = { cwd: tmpDir };

  await t.test('write_file creates file correctly', async () => {
    const res = await registry.execute(
      'write_file',
      JSON.stringify({ path: 'test.txt', content: 'hello world\nline 2\nline 3' }),
      'call-1',
      context
    );
    assert.strictEqual(res.isError, false);
    assert.match(res.result, /Successfully wrote/);

    const content = await fs.readFile(path.join(tmpDir, 'test.txt'), 'utf8');
    assert.strictEqual(content, 'hello world\nline 2\nline 3');
  });

  await t.test('view_file reads line numbers', async () => {
    const res = await registry.execute(
      'view_file',
      JSON.stringify({ path: 'test.txt', offset: 1, limit: 2 }),
      'call-2',
      context
    );
    assert.strictEqual(res.isError, false);
    assert.match(res.result, /1 \| hello world/);
    assert.match(res.result, /2 \| line 2/);
  });

  await t.test('edit_file replaces text and returns unified diff', async () => {
    const res = await registry.execute(
      'edit_file',
      JSON.stringify({ path: 'test.txt', target: 'line 2', replacement: 'line 2 edited' }),
      'call-3',
      context
    );
    assert.strictEqual(res.isError, false);
    assert.ok(res.diff, 'Expected diff in edit result');
    assert.match(res.diff, /-line 2/);
    assert.match(res.diff, /\+line 2 edited/);

    const diffRendered = renderDiff(res.diff);
    assert.ok(diffRendered.length > 0);

    const updated = await fs.readFile(path.join(tmpDir, 'test.txt'), 'utf8');
    assert.strictEqual(updated, 'hello world\nline 2 edited\nline 3');
  });

  await t.test('list_dir lists created file', async () => {
    const res = await registry.execute(
      'list_dir',
      JSON.stringify({ path: '.' }),
      'call-4',
      context
    );
    assert.strictEqual(res.isError, false);
    assert.match(res.result, /test\.txt/);
  });

  await t.test('grep_search finds pattern', async () => {
    const res = await registry.execute(
      'grep_search',
      JSON.stringify({ query: 'edited' }),
      'call-5',
      context
    );
    assert.strictEqual(res.isError, false);
    assert.match(res.result, /test\.txt:2: line 2 edited/);
  });

  await t.test('bash runs command in workspace', async () => {
    const res = await registry.execute(
      'bash',
      JSON.stringify({ command: 'echo "bash tool works"' }),
      'call-6',
      context
    );
    assert.strictEqual(res.isError, false);
    assert.match(res.result, /bash tool works/);
  });

  // Cleanup
  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('bash runs autonomously and stays controllable', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-bash-'));
  const ctx = (extra: object = {}) => ({ cwd: tmpDir, ...extra });

  await t.test('commands run without any approval step', async () => {
    const result = await bashTool.execute({ command: 'echo autonomous' }, ctx());
    assert.match(result, /autonomous/);
  });

  await t.test('a backgrounded server does not block the turn', async () => {
    const started = Date.now();
    const result = await bashTool.execute(
      { command: 'sleep 8 & echo "listening on :3000"', timeoutMs: 15_000 },
      ctx()
    );
    assert.match(result, /listening on :3000/);
    assert.ok(Date.now() - started < 3000, 'returned while the background process was still running');
  });

  await t.test('abort kills the whole process group, descendants included', async () => {
    const marker = path.join(tmpDir, 'survivor');
    const controller = new AbortController();
    const run = bashTool.execute(
      { command: `(sleep 2; echo alive > ${marker}) & sleep 5` },
      ctx({ abortSignal: controller.signal })
    );
    setTimeout(() => controller.abort(), 250);
    await run;
    await new Promise((resolve) => setTimeout(resolve, 2600));
    assert.strictEqual(fsSync.existsSync(marker), false, 'a descendant outlived the abort');
  });

  await t.test('output keeps the end and spills the whole log to a file', async () => {
    const result = await bashTool.execute({ command: 'seq 1 200000' }, ctx({ outputLimitBytes: 4096 }));
    assert.match(result, /\b200000\b/, 'the end of the output is what the model sees');
    assert.strictEqual(/\n1\n/.test(result), false, 'the start was dropped, not the end');

    const spill = /Full log: (\S+?)\]/.exec(result)?.[1];
    assert.ok(spill && fsSync.existsSync(spill), 'the complete log is on disk for the model to read');
    assert.ok(fsSync.statSync(spill).size > 1_000_000, 'the spilled log holds the full output');
    fsSync.rmSync(spill, { force: true });
  });

  await t.test('the output cap follows the model context window', async () => {
    assert.ok(toolOutputLimitBytes(22480) < toolOutputLimitBytes(131072));
    assert.strictEqual(toolOutputLimitBytes(4_000_000), 50 * 1024, 'a huge window still has a ceiling');
    assert.strictEqual(toolOutputLimitBytes(1), 4 * 1024, 'a tiny window still gets a usable floor');
  });

  await t.test('a command waiting on stdin cannot hang the agent', async () => {
    const result = await bashTool.execute({ command: 'read -r line; echo "got:$line"' }, ctx());
    assert.match(result, /got:/);
  });

  await t.test('exit status and stderr reach the model', async () => {
    const result = await bashTool.execute({ command: 'echo boom >&2; exit 3' }, ctx());
    assert.match(result, /boom/);
    assert.match(result, /exited with code 3/);
  });

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('edit_file writes the replacement literally', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-edit-'));
  const file = path.join(tmpDir, 'sample.sh');

  // $&, $`, $' and $n are regex substitution patterns. A replacement carrying them
  // must land in the file byte for byte.
  const cases: Array<[string, string]> = [
    ['shell positional', 'run() { grep -o "x" <<< "$&"; }'],
    ['backtick pattern', 'echo "$` and $\' done"'],
    ['regex group ref', "line.replace(/(a)(b)/, '$2$1')"],
    ['jquery-ish', 'const $1 = $("#id"); // $& $$'],
  ];

  for (const [label, replacement] of cases) {
    await fs.writeFile(file, 'PLACEHOLDER\n', 'utf8');
    const result = await editFileTool.execute({ path: file, target: 'PLACEHOLDER', replacement }, { cwd: tmpDir });
    assert.strictEqual(result.isError, false, label);
    assert.strictEqual((await fs.readFile(file, 'utf8')).trimEnd(), replacement, label);
  }

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('A malformed step limit fails loudly instead of muting the agent', async (t) => {
  await t.test('a non-numeric limit is rejected, not silently applied', () => {
    for (const bad of [Number.NaN, -1, 2.5]) {
      assert.throws(() => resolveMaxSteps(bad), /non-negative integer/);
    }
    assert.throws(() => resolveMaxSteps(undefined, 'lots'), /DSH_MAX_STEPS/);
  });

  await t.test('valid limits still resolve, with 0 meaning unlimited', () => {
    assert.strictEqual(resolveMaxSteps(5), 5);
    assert.strictEqual(resolveMaxSteps(0), Infinity);
    assert.strictEqual(resolveMaxSteps(undefined, '7'), 7);
    assert.strictEqual(resolveMaxSteps(undefined, undefined), 100);
    assert.strictEqual(resolveMaxSteps(undefined, '  '), 100, 'an empty env var is not a limit');
  });

  await t.test('the agent refuses to construct with a limit that would mute it', () => {
    const client = { hasModel: () => true, streamChat: async () => ({ content: '', toolCalls: [] }) } as never;
    assert.throws(() => new Agent({ client, maxSteps: Number.NaN }), /non-negative integer/);
  });
});

test('Search tools stay responsive and agree on what to skip', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-search-'));
  await fs.mkdir(path.join(tmpDir, 'lib'), { recursive: true });
  await fs.mkdir(path.join(tmpDir, '.dsh', 'sessions'), { recursive: true });
  await fs.mkdir(path.join(tmpDir, 'node_modules'), { recursive: true });
  await fs.writeFile(path.join(tmpDir, 'lib', 'core.ts'), 'export const NEEDLE = 1;\n');
  await fs.writeFile(path.join(tmpDir, '.dsh', 'sessions', 's.jsonl'), '{"content":"NEEDLE"}\n');
  await fs.writeFile(path.join(tmpDir, 'node_modules', 'dep.js'), 'const NEEDLE = 2;\n');

  await t.test('lib/ is searched; it holds source in plenty of projects', async () => {
    const result = await grepSearchTool.execute({ query: 'NEEDLE' }, { cwd: tmpDir });
    assert.match(result, /lib\/core\.ts/);
  });

  await t.test("the agent's own session log is not searched", async () => {
    const result = await grepSearchTool.execute({ query: 'NEEDLE' }, { cwd: tmpDir });
    assert.doesNotMatch(result, /\.dsh/, 'transcripts would match everything the conversation said');
    assert.doesNotMatch(result, /node_modules/);
  });

  await t.test('grep_search stops when the turn is aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const result = await grepSearchTool.execute({ query: 'NEEDLE' }, { cwd: tmpDir, abortSignal: controller.signal });
    assert.match(result, /cancelled by user/);
  });

  await t.test('list_dir stops when the turn is aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const result = await listDirTool.execute({ recursive: true }, { cwd: tmpDir, abortSignal: controller.signal });
    assert.match(result, /cancelled by user/);
  });

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('view_file serves a window without loading the whole file', async (t) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-view-'));

  await t.test('a slice deep inside a large file returns only that slice', async () => {
    const big = path.join(tmpDir, 'big.log');
    const handle = await fs.open(big, 'w');
    for (let block = 0; block < 40; block++) {
      await handle.write(Array.from({ length: 5000 }, (_, i) => `line ${block * 5000 + i + 1}`).join('\n') + '\n');
    }
    await handle.close();
    assert.ok((await fs.stat(big)).size > 2_000_000, 'fixture should be large');

    // Reading line by line is what bounds memory; what is observable here is that a deep slice
    // comes back correctly and reports that the file continues, rather than the whole file.
    const result = await viewFileTool.execute({ path: big, offset: 150_000, limit: 3 }, { cwd: tmpDir });

    assert.match(result, /150000 \| line 150000/);
    assert.match(result, /150002 \| line 150002/);
    assert.doesNotMatch(result, /line 150003/, 'the window stops at the requested limit');
    assert.doesNotMatch(result, /line 1\b/, 'earlier lines are not returned');
    assert.match(result, /more follow/);
  });

  await t.test('a binary file is reported, not spilled into the context', async () => {
    const bin = path.join(tmpDir, 'image.png');
    await fs.writeFile(bin, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00, 0x00, 0x01, 0x02, 0x00, 0xff]));
    const result = await viewFileTool.execute({ path: bin }, { cwd: tmpDir });
    assert.match(result, /binary file/);
    assert.doesNotMatch(result, /\|/, 'no line-numbered garbage');
  });

  await t.test('a directory says so instead of failing obscurely', async () => {
    assert.match(await viewFileTool.execute({ path: '.' }, { cwd: tmpDir }), /is a directory/);
  });

  await t.test('an offset past the end reports the length', async () => {
    const small = path.join(tmpDir, 'small.txt');
    await fs.writeFile(small, 'a\nb\nc\n');
    assert.match(await viewFileTool.execute({ path: small, offset: 99 }, { cwd: tmpDir }), /past the end/);
  });

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('bash refuses to empty the workspace or reach outside it', async (t) => {
  const cwd = '/Users/example/project';

  await t.test('the shapes that destroy a workspace', () => {
    for (const command of [
      'rm -rf /Users/example/project/*',
      'rm -rf ./*',
      'rm -rf .',
      'rm -rf ~',
      'rm -rf /',
      'rm -rf /Users/example',
      'find . -delete',
      'git reset --hard',
      'git clean -fdx',
      'dd if=/dev/zero of=/dev/disk0',
      'npm test && rm -rf /Users/example/project/*',
    ]) {
      assert.strictEqual(judgeCommand(command, cwd).refused, true, `should refuse: ${command}`);
    }
  });

  await t.test('ordinary work is untouched', () => {
    for (const command of [
      'rm -rf node_modules',
      'rm -rf build/*',
      'rm -f src/old.ts',
      'rm -rf dist && npm run build',
      'npm test',
      'git status',
      'git commit -m "wip"',
      'python3 game.py',
      'mkdir -p game/{css,js}',
      'cat package.json',
    ]) {
      assert.strictEqual(judgeCommand(command, cwd).refused, false, `should allow: ${command}`);
    }
  });

  await t.test('the refusal says what to do instead', () => {
    const verdict = judgeCommand('rm -rf ./*', cwd);
    const message = refusalMessage(verdict.reason!);
    assert.match(message, /Name the specific files/);
    assert.match(message, /\/clear/, 'the model should be pointed at the command that was meant');
  });

  await t.test('a refused command does not run', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dsh-guard-'));
    await fs.writeFile(path.join(tmpDir, 'keep.txt'), 'still here\n');

    const result = await bashTool.execute({ command: 'rm -rf ./*' }, { cwd: tmpDir });
    assert.match(result, /Refused/);
    assert.strictEqual(await fs.readFile(path.join(tmpDir, 'keep.txt'), 'utf8'), 'still here\n');

    await fs.rm(tmpDir, { recursive: true, force: true });
  });
});

test('the session history is not something a command may delete', () => {
  const cwd = '/Users/example/project';
  for (const command of [
    'rm -rf .dsh',
    'rm -rf .dsh/sessions',
    'rm -f .dsh/sessions/session-1.jsonl',
    'rm -rf ./.dsh/sessions/*',
  ]) {
    assert.strictEqual(judgeCommand(command, cwd).refused, true, `should refuse: ${command}`);
  }
  // A directory that merely mentions it is fine.
  assert.strictEqual(judgeCommand('rm -rf dsh-docs', cwd).refused, false);
});
