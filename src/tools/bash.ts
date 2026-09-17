import { randomBytes } from 'node:crypto';
import { createWriteStream, type WriteStream } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { ToolDefinition } from './types.js';
import { runShellCommand } from './child-process.js';
import { toolOutputLimitBytes } from './limits.js';
import { judgeCommand, refusalMessage } from './destructive.js';

interface BashArgs {
  command: string;
  timeoutMs?: number;
}

const DEFAULT_TIMEOUT_MS = 120_000;

/**
 * Collects command output in bounded memory, keeping the end rather than the beginning: a failing
 * build puts its errors last. Once the output outgrows the limit the complete text goes to a temp
 * file, which the result names so the model can read the rest with view_file or grep.
 */
class OutputTail {
  private chunks: Buffer[] = [];
  private totalBytes = 0;
  private stream?: WriteStream;
  private path?: string;

  constructor(private readonly maxBytes: number) {}

  public append(data: Buffer): void {
    this.totalBytes += data.length;

    if (!this.stream && this.totalBytes > this.maxBytes) {
      this.path = join(tmpdir(), `dsh-output-${randomBytes(6).toString('hex')}.log`);
      this.stream = createWriteStream(this.path);
      for (const chunk of this.chunks) this.stream.write(chunk);
    }

    if (this.stream) {
      this.stream.write(data);
    }

    this.chunks.push(data);
    this.trimToTail();
  }

  private trimToTail(): void {
    let held = this.chunks.reduce((acc, c) => acc + c.length, 0);
    while (this.chunks.length > 1 && held - this.chunks[0].length >= this.maxBytes) {
      held -= this.chunks.shift()!.length;
    }
  }

  public async finish(): Promise<{ text: string; truncated: boolean; fullOutputPath?: string }> {
    const stream = this.stream;
    if (stream) {
      await new Promise<void>((resolve) => stream.end(resolve));
    }

    let text = Buffer.concat(this.chunks).toString('utf8');
    if (text.length > this.maxBytes) text = text.slice(text.length - this.maxBytes);

    return {
      text,
      truncated: this.stream !== undefined,
      fullOutputPath: this.path,
    };
  }
}

export const bashTool: ToolDefinition<BashArgs, string> = {
  name: 'bash',
  description:
    'Execute a shell command in the current workspace directory. Use this for running tests, build tools, git, servers, or inspecting the environment. Returns stdout and stderr; long output keeps the end and saves the whole log to a file.',
  parameters: {
    type: 'object',
    properties: {
      command: {
        type: 'string',
        description: 'The exact bash command line to run.',
      },
      timeoutMs: {
        type: 'number',
        description: 'Optional execution timeout in milliseconds (default: 120000).',
      },
    },
    required: ['command'],
  },
  execute: async ({ command, timeoutMs = DEFAULT_TIMEOUT_MS }, context) => {
    if (context.abortSignal?.aborted) {
      return 'Execution cancelled by user.';
    }

    const verdict = judgeCommand(command, context.cwd);
    if (verdict.refused) {
      return refusalMessage(verdict.reason!);
    }

    const limitBytes = context.outputLimitBytes ?? toolOutputLimitBytes(32768);
    const output = new OutputTail(limitBytes);

    let exitCode: number | null = 0;
    let timedOut = false;
    let spawnError: string | undefined;

    try {
      const result = await runShellCommand(command, context.cwd, {
        signal: context.abortSignal,
        timeoutMs,
        onData: (data) => output.append(data),
      });
      exitCode = result.exitCode;
      timedOut = result.timedOut;
    } catch (err: any) {
      spawnError = err?.message ?? String(err);
    }

    const { text, truncated, fullOutputPath } = await output.finish();
    const parts: string[] = [];
    if (text.trimEnd()) parts.push(text.trimEnd());
    if (truncated && fullOutputPath) {
      parts.push(`[Output truncated to the last ${limitBytes} bytes. Full log: ${fullOutputPath}]`);
    }

    if (context.abortSignal?.aborted) {
      parts.push('Command cancelled by user.');
    } else if (spawnError) {
      parts.push(`Command could not start: ${spawnError}`);
    } else if (timedOut) {
      parts.push(`Command timed out after ${timeoutMs}ms.`);
    } else if (exitCode !== 0) {
      parts.push(`Command exited with code ${exitCode ?? 'unknown (killed by a signal)'}.`);
    }

    return parts.join('\n\n') || '(command completed with no output)';
  },
};
