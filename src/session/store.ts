import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { randomUUID } from 'node:crypto';
import type { ChatMessage } from '../llm/types.js';

export interface SessionMetadata {
  id: string;
  createdAt: string;
  updatedAt: string;
  sizeBytes: number;
  /** Opening words of the first user message, for telling sessions apart. */
  preview?: string;
}

/** Bytes read from the head of a log to recover its first user message. */
const PREVIEW_BYTES = 4096;

export class SessionStore {
  private sessionsDir: string;
  private initialized = false;

  constructor(workspaceCwd?: string) {
    if (workspaceCwd) {
      this.sessionsDir = path.join(workspaceCwd, '.dsh', 'sessions');
    } else {
      this.sessionsDir = path.join(os.homedir(), '.dsh', 'sessions');
    }
  }

  /**
   * Creates the session directory once per store, and marks it ignored by git. Transcripts land in
   * whatever repository dsh was started in, where they would otherwise show up as untracked files
   * and be easy to commit by accident; a `.gitignore` of `*` inside makes the whole tree invisible
   * without touching the repository's own ignore rules.
   */
  public async init(): Promise<void> {
    if (this.initialized) return;
    const root = path.dirname(this.sessionsDir);
    await fs.mkdir(this.sessionsDir, { recursive: true });
    await fs.writeFile(path.join(root, '.gitignore'), '*\n', { flag: 'w' });
    this.initialized = true;
  }

  public createSessionId(): string {
    const timestamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
    const shortId = randomUUID().slice(0, 8);
    return `session-${timestamp}-${shortId}`;
  }

  private getSessionPath(sessionId: string): string {
    return path.join(this.sessionsDir, `${sessionId}.jsonl`);
  }

  public async appendMessage(sessionId: string, message: ChatMessage): Promise<void> {
    await this.init();
    const filePath = this.getSessionPath(sessionId);
    const line = JSON.stringify({
      timestamp: new Date().toISOString(),
      ...message,
    }) + '\n';
    await fs.appendFile(filePath, line, 'utf8');
  }

  public async loadSession(sessionId: string): Promise<ChatMessage[]> {
    const filePath = this.getSessionPath(sessionId);
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const lines = content.split('\n').filter(l => l.trim().length > 0);
      return lines.map(line => {
        const parsed = JSON.parse(line);
        return {
          role: parsed.role,
          content: parsed.content,
          name: parsed.name,
          tool_call_id: parsed.tool_call_id,
          tool_calls: parsed.tool_calls,
          reasoning_content: parsed.reasoning_content,
        };
      });
    } catch (err: any) {
      throw new Error(`Failed to load session ${sessionId}: ${err.message}`);
    }
  }

  public async listSessions(): Promise<SessionMetadata[]> {
    await this.init();
    try {
      const files = await fs.readdir(this.sessionsDir);
      const jsonlFiles = files.filter(f => f.endsWith('.jsonl'));

      const sessions: SessionMetadata[] = [];
      for (const file of jsonlFiles) {
        const filePath = path.join(this.sessionsDir, file);
        try {
          const stat = await fs.stat(filePath);
          sessions.push({
            id: file.replace(/\.jsonl$/, ''),
            createdAt: stat.birthtime.toISOString(),
            updatedAt: stat.mtime.toISOString(),
            sizeBytes: stat.size,
            preview: await this.readPreview(filePath),
          });
        } catch (err) {
          void err; // Unreadable or deleted mid-scan; the rest of the list still stands.
        }
      }

      sessions.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
      return sessions;
    } catch {
      return [];
    }
  }

  /**
   * First user message of a log, read from the head of the file. Listing must not cost the size of
   * every transcript, so the whole file is never read just to describe it.
   */
  private async readPreview(filePath: string): Promise<string | undefined> {
    const handle = await fs.open(filePath, 'r');
    try {
      const buffer = Buffer.alloc(PREVIEW_BYTES);
      const { bytesRead } = await handle.read(buffer, 0, PREVIEW_BYTES, 0);
      for (const line of buffer.subarray(0, bytesRead).toString('utf8').split('\n')) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line);
          if (parsed.role === 'user' && typeof parsed.content === 'string') {
            const text = parsed.content.replace(/\s+/g, ' ').trim();
            return text.length > 80 ? `${text.slice(0, 80)}…` : text;
          }
        } catch (err) {
          void err; // A line split by the read window; earlier lines already had their chance.
        }
      }
      return undefined;
    } finally {
      await handle.close();
    }
  }
}
