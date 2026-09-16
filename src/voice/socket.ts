import * as fs from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';

/** Where a voice daemon connects to drive this harness. */
export const VOICE_SOCKET_PATH = path.join(os.homedir(), '.dsh', 'input.sock');

/** What a daemon may ask the harness to do. Speech is classified before it gets here. */
export type VoiceCommand =
  | { type: 'text'; text: string }
  | { type: 'submit' }
  | { type: 'abort' };

/** What the harness tells the daemon, for it to speak. */
export type VoiceEvent =
  | { type: 'ack' }
  | { type: 'done'; summary: string }
  | { type: 'error'; message: string };

export interface VoiceSocketHandlers {
  onText(text: string): void;
  onSubmit(): void;
  onAbort(): void;
}

/** The longest line a client may send, so a runaway writer cannot exhaust memory. */
const MAX_LINE_BYTES = 64 * 1024;

/**
 * Parses one line from a client into a command.
 *
 * Returns a string describing the problem instead of throwing, so a malformed line is answered and
 * the connection stays up: a daemon mid-development should get told what it sent wrong.
 */
export function parseVoiceCommand(line: string): VoiceCommand | string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    return 'not valid JSON';
  }
  if (typeof parsed !== 'object' || parsed === null) return 'expected a JSON object';

  const { type, text } = parsed as { type?: unknown; text?: unknown };
  if (type === 'submit' || type === 'abort') return { type };
  if (type === 'text') {
    if (typeof text !== 'string') return 'text commands need a "text" string';
    return { type: 'text', text };
  }
  return `unknown type ${JSON.stringify(type)}; expected text, submit or abort`;
}

/**
 * Condenses a reply into something short enough to speak.
 *
 * Voice mode's system prompt asks the model for one or two sentences; this is the backstop for when
 * it answers at length anyway, so the daemon never reads a page aloud.
 */
export function summariseForSpeech(reply: string, maxSentences = 2, maxChars = 320): string {
  const flat = reply.replace(/```[\s\S]*?```/g, ' ').replace(/\s+/g, ' ').trim();
  if (!flat) return 'Done.';

  const sentences = flat.match(/[^.!?]+[.!?]+|\S[^.!?]*$/g) ?? [flat];
  let out = sentences
    .slice(0, maxSentences)
    .map((sentence) => sentence.trim())
    .join(' ');
  if (out.length > maxChars) out = `${out.slice(0, maxChars - 1).trimEnd()}…`;
  return out;
}

/**
 * A local listener that lets a voice daemon type into this harness.
 *
 * The socket is a unix domain socket with owner-only permissions rather than a TCP port: anything
 * that can write to it can make the agent act, so it must not be reachable from off the machine.
 */
export class VoiceSocket {
  private server?: net.Server;
  private readonly clients = new Set<net.Socket>();

  constructor(private readonly socketPath: string = VOICE_SOCKET_PATH) {}

  public get isOpen(): boolean {
    return this.server !== undefined;
  }

  public get address(): string {
    return this.socketPath;
  }

  public async open(handlers: VoiceSocketHandlers): Promise<void> {
    if (this.server) return;
    await fs.promises.mkdir(path.dirname(this.socketPath), { recursive: true });
    await this.clearStaleSocket();

    const server = net.createServer((socket) => this.attachClient(socket, handlers));
    this.server = server;

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(this.socketPath, () => {
        server.removeListener('error', reject);
        resolve();
      });
    }).catch((err) => {
      this.server = undefined;
      throw err;
    });

    // listen() applies the umask, which usually leaves the socket group- and world-writable.
    await fs.promises.chmod(this.socketPath, 0o600);
  }

  /**
   * Removes a socket file left behind by a harness that did not shut down cleanly. A socket someone
   * is still listening on is left alone, so two instances cannot silently steal each other's voice.
   */
  private async clearStaleSocket(): Promise<void> {
    if (!fs.existsSync(this.socketPath)) return;

    const inUse = await new Promise<boolean>((resolve) => {
      const probe = net.connect(this.socketPath);
      probe.once('connect', () => {
        probe.destroy();
        resolve(true);
      });
      probe.once('error', () => {
        probe.destroy();
        resolve(false);
      });
    });

    if (inUse) {
      throw new Error(`${this.socketPath} is already in use by another dsh. Leave voice mode there first.`);
    }
    await fs.promises.unlink(this.socketPath);
  }

  private attachClient(socket: net.Socket, handlers: VoiceSocketHandlers): void {
    this.clients.add(socket);
    socket.setEncoding('utf8');

    let buffer = '';
    socket.on('data', (chunk: string) => {
      buffer += chunk;
      if (buffer.length > MAX_LINE_BYTES) {
        buffer = '';
        this.send(socket, { type: 'error', message: 'line too long' });
        return;
      }

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const command = parseVoiceCommand(line);
        if (typeof command === 'string') {
          this.send(socket, { type: 'error', message: command });
          continue;
        }
        if (command.type === 'text') handlers.onText(command.text);
        else if (command.type === 'submit') handlers.onSubmit();
        else handlers.onAbort();
      }
    });

    const drop = () => {
      this.clients.delete(socket);
    };
    socket.on('close', drop);
    socket.on('error', drop);
  }

  private send(socket: net.Socket, event: VoiceEvent): void {
    if (socket.destroyed) return;
    socket.write(`${JSON.stringify(event)}\n`);
  }

  /** Tells every connected daemon what just happened. */
  public broadcast(event: VoiceEvent): void {
    for (const socket of this.clients) this.send(socket, event);
  }

  public async close(): Promise<void> {
    const server = this.server;
    this.server = undefined;
    if (!server) return;

    for (const socket of this.clients) socket.destroy();
    this.clients.clear();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await fs.promises.unlink(this.socketPath).catch((err) => void err);
  }
}
