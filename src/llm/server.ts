import { spawn, type ChildProcess } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import type { LiteModel } from '../config/models.js';
import { killRunningCommands } from '../tools/child-process.js';

export interface LlamaServerManagerOptions {
  logFile?: string;
  readyTimeoutMs?: number;
  pollIntervalMs?: number;
}

export type HealthState = 'ready' | 'loading' | 'foreign' | 'down';

export function serverOrigin(baseUrl: string): string {
  const url = new URL(baseUrl);
  return `${url.protocol}//${url.host}`;
}

export function serverPort(baseUrl: string): string {
  const url = new URL(baseUrl);
  return url.port || (url.protocol === 'https:' ? '443' : '80');
}

export function getLocalIpAddress(): string {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const net of interfaces[name] || []) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return '127.0.0.1';
}

export function buildServerArgs(model: LiteModel, isHost = false): string[] {
  const args = ['-m', model.modelPath, ...model.launchArgs];
  if (!args.includes('--port') && !args.includes('-p')) {
    args.push('--port', serverPort(model.baseUrl));
  }
  if (!args.includes('--timeout') && !args.includes('-to')) {
    args.push('--timeout', '86400');
  }
  if (isHost && !args.includes('--host') && !args.includes('-H')) {
    args.push('--host', '0.0.0.0');
  }
  return args;
}

function isRunning(child: ChildProcess): boolean {
  return child.pid !== undefined && child.exitCode === null && child.signalCode === null;
}

/**
 * How a dead child died. A child killed by a signal — an OOM kill while
 * loading a large model, say — reports exitCode null and carries signalCode
 * instead, so exitCode alone cannot tell death from still-loading.
 */
function describeExit(child: ChildProcess): string {
  return child.signalCode !== null ? `signal ${child.signalCode}` : `code ${child.exitCode}`;
}

function samePath(a: string, b: string): boolean {
  try {
    return fs.realpathSync(a) === fs.realpathSync(b);
  } catch {
    return path.resolve(a) === path.resolve(b);
  }
}

/**
 * Whether a server launched for `running` can serve `wanted` as-is.
 * Entries that share a GGUF still differ in --ctx-size or --mmproj, so the
 * weights path alone is not enough to decide reuse. --host is excluded: it
 * changes who can reach the server, not what the server does with a request.
 */
export function sameLaunch(running: LiteModel, wanted: LiteModel): boolean {
  return (
    samePath(running.modelPath, wanted.modelPath) &&
    running.llamaServer === wanted.llamaServer &&
    serverOrigin(running.baseUrl) === serverOrigin(wanted.baseUrl) &&
    JSON.stringify(buildServerArgs(running)) === JSON.stringify(buildServerArgs(wanted))
  );
}

export class LlamaServerManager {
  private readonly logFile: string;
  private readonly readyTimeoutMs: number;
  private readonly pollIntervalMs: number;
  private child?: ChildProcess;
  private active?: LiteModel;
  private logOffset = 0;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(options: LlamaServerManagerOptions = {}) {
    this.logFile = options.logFile || path.join(os.homedir(), '.dsh', 'logs', 'llama-server.log');
    this.readyTimeoutMs = options.readyTimeoutMs || 5 * 60_000;
    this.pollIntervalMs = options.pollIntervalMs || 300;
  }

  public get model(): LiteModel | undefined {
    return this.active;
  }

  public get isRunning(): boolean {
    return this.child !== undefined && isRunning(this.child);
  }

  /**
   * Runs lifecycle operations one at a time. They spawn and kill a shared
   * child, so overlapping calls would tear down each other's server: a switch
   * still loading when a prompt is submitted used to leave the model selected
   * but nothing listening. Internal callers use the unserialized *Now methods.
   */
  private serialize<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.queue.then(operation, operation);
    this.queue = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  }

  public ensure(
    model: LiteModel,
    onStatus?: (msg: string) => void,
    abortSignal?: AbortSignal
  ): Promise<void> {
    return this.serialize(() => this.ensureNow(model, onStatus, abortSignal));
  }

  private async ensureNow(
    model: LiteModel,
    onStatus?: (msg: string) => void,
    abortSignal?: AbortSignal
  ): Promise<void> {
    const origin = serverOrigin(model.baseUrl);

    // If already active with the same launch configuration and healthy, reuse
    if (this.active && sameLaunch(this.active, model) && (await this.health(origin, abortSignal)) === 'ready') {
      this.active = model;
      return;
    }

    await this.stopNow();

    const state = await this.health(origin, abortSignal);
    if (state === 'foreign') {
      throw new Error(`${origin} is currently in use by another service. Free the port or change baseUrl.`);
    }

    if (state !== 'down') {
      onStatus?.(`Found existing server at ${origin}, checking model...`);
      if (state === 'loading') {
        await this.waitUntilReady(origin, undefined, abortSignal);
      }
      const servedPath = await this.servedModelPath(origin, abortSignal);
      if (servedPath && samePath(servedPath, model.modelPath)) {
        onStatus?.(`Reusing active llama-server serving ${model.name}`);
        this.active = model;
        return;
      }
      throw new Error(
        `llama-server at ${origin} is serving ${servedPath || 'another model'}, not ${model.name}. ` +
          `Stop that server or change port in launchArgs.`
      );
    }

    onStatus?.(`Starting llama-server for ${model.name}...`);
    const child = this.spawnServer(model);
    this.child = child;

    try {
      await this.waitUntilReady(origin, child, abortSignal);
      onStatus?.(`llama-server is ready for ${model.name}`);
    } catch (err) {
      await this.stopNow();
      throw err;
    }

    this.active = model;
  }

  public stop(): Promise<void> {
    return this.serialize(() => this.stopNow());
  }

  private async stopNow(): Promise<void> {
    const child = this.child;
    this.child = undefined;
    this.active = undefined;
    if (!child || !isRunning(child)) return;

    const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()));
    child.kill('SIGTERM');
    const forceKill = setTimeout(() => child.kill('SIGKILL'), 3000);
    await exited;
    clearTimeout(forceKill);
  }

  public stopSync(): void {
    if (this.child && isRunning(this.child)) {
      this.child.kill('SIGTERM');
    }
    this.child = undefined;
    this.active = undefined;
  }

  public startHost(
    model: LiteModel,
    onLogLine: (line: string) => void,
    abortSignal?: AbortSignal
  ): Promise<{ port: string; localIp: string; localUrl: string; remoteUrl: string }> {
    return this.serialize(() => this.startHostNow(model, onLogLine, abortSignal));
  }

  private async startHostNow(
    model: LiteModel,
    onLogLine: (line: string) => void,
    abortSignal?: AbortSignal
  ): Promise<{ port: string; localIp: string; localUrl: string; remoteUrl: string }> {
    await this.stopNow();
    const port = serverPort(model.baseUrl);
    const origin = `http://localhost:${port}`;

    const child = this.spawnServer(model, true, onLogLine);
    this.child = child;

    try {
      await this.waitUntilReady(origin, child, abortSignal);
    } catch (err) {
      await this.stopNow();
      throw err;
    }

    this.active = model;
    const localIp = getLocalIpAddress();
    return {
      port,
      localIp,
      localUrl: `http://localhost:${port}/v1`,
      remoteUrl: `http://${localIp}:${port}/v1`,
    };
  }

  private spawnServer(
    model: LiteModel,
    isHost = false,
    onLogLine?: (line: string) => void
  ): ChildProcess {
    const args = buildServerArgs(model, isHost);
    fs.mkdirSync(path.dirname(this.logFile), { recursive: true });
    const logStream = fs.createWriteStream(this.logFile, { flags: 'a' });

    logStream.write(`\n=== ${new Date().toISOString()} ${model.llamaServer} ${args.join(' ')}\n`);

    const child = spawn(model.llamaServer, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdoutBuf = '';
    child.stdout?.on('data', (chunk: Buffer) => {
      logStream.write(chunk);
      if (onLogLine) {
        stdoutBuf += chunk.toString('utf8');
        const lines = stdoutBuf.split('\n');
        stdoutBuf = lines.pop() || '';
        for (const line of lines) {
          if (line.trim()) onLogLine(line);
        }
      }
    });

    let stderrBuf = '';
    child.stderr?.on('data', (chunk: Buffer) => {
      logStream.write(chunk);
      if (onLogLine) {
        stderrBuf += chunk.toString('utf8');
        const lines = stderrBuf.split('\n');
        stderrBuf = lines.pop() || '';
        for (const line of lines) {
          if (line.trim()) onLogLine(line);
        }
      }
    });

    child.on('close', () => {
      logStream.end();
    });

    return child;
  }

  private async waitUntilReady(
    origin: string,
    child: ChildProcess | undefined,
    signal?: AbortSignal
  ): Promise<void> {
    const deadline = Date.now() + this.readyTimeoutMs;

    while (true) {
      signal?.throwIfAborted();
      if (child && !isRunning(child)) {
        throw new Error(`llama-server exited unexpectedly with ${describeExit(child)}.${this.logTail()}`);
      }

      if ((await this.health(origin, signal)) === 'ready') return;

      if (Date.now() >= deadline) {
        const secs = Math.round(this.readyTimeoutMs / 1000);
        throw new Error(`llama-server at ${origin} was not ready after ${secs}s.${this.logTail()}`);
      }

      await new Promise((r) => setTimeout(r, this.pollIntervalMs));
    }
  }

  private async health(origin: string, signal?: AbortSignal): Promise<HealthState> {
    const timeout = AbortSignal.timeout(2000);
    try {
      const response = await fetch(`${origin}/health`, {
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
      if (response.ok) return 'ready';
      return response.status === 503 ? 'loading' : 'foreign';
    } catch {
      signal?.throwIfAborted();
      return 'down';
    }
  }

  private async servedModelPath(origin: string, signal?: AbortSignal): Promise<string | undefined> {
    try {
      const res = await fetch(`${origin}/props`, { signal });
      if (!res.ok) return undefined;
      const props = (await res.json()) as any;
      return typeof props.model_path === 'string' ? props.model_path : undefined;
    } catch {
      return undefined;
    }
  }

  private logTail(): string {
    try {
      const fd = fs.openSync(this.logFile, 'r');
      try {
        const size = fs.fstatSync(fd).size;
        const start = Math.max(this.logOffset, size - 8192);
        const buffer = Buffer.alloc(size - start);
        fs.readSync(fd, buffer, 0, buffer.length, start);
        const lines = buffer.toString('utf8').trimEnd().split('\n').slice(-15);
        return lines.length > 0 ? `\n--- ${this.logFile} ---\n${lines.join('\n')}` : '';
      } finally {
        fs.closeSync(fd);
      }
    } catch {
      return '';
    }
  }
}

/**
 * Tears down everything this process started when it exits: the llama-server and any shell command
 * still running. Commands run in detached process groups, so without this they outlive the harness.
 */
export function stopServerOnExit(manager: LlamaServerManager): void {
  const teardown = () => {
    manager.stopSync();
    killRunningCommands();
  };

  process.once('exit', teardown);
  const signals = ['SIGINT', 'SIGTERM', 'SIGHUP'] as const;
  for (const sig of signals) {
    process.once(sig, () => {
      teardown();
      process.exit(0);
    });
  }
}
