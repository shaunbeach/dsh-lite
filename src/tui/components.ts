import { homedir } from 'node:os';
import {
  type Component,
  Container,
  Markdown,
  Spacer,
  stripTerminalSequences,
  Text,
  truncateToWidth,
  visibleWidth,
  wrapTextWithAnsi,
} from '@earendil-works/pi-tui';
import { markdownTheme, style } from './theme.js';

const THINKING_WINDOW_LINES = 6;
const THINKING_TAIL_CHARS = 4000;
const TOOL_PREVIEW_LINES = 6;
const DIFF_PREVIEW_LINES = 16;

export class Line implements Component {
  private text: string;
  private cached?: { width: number; lines: string[] };

  constructor(text = '') {
    this.text = text;
  }

  setText(text: string): void {
    if (text === this.text) return;
    this.text = text;
    this.cached = undefined;
  }

  invalidate(): void {
    this.cached = undefined;
  }

  render(width: number): string[] {
    if (this.cached?.width !== width) {
      this.cached = { width, lines: this.text ? [truncateToWidth(this.text, width)] : [] };
    }
    return this.cached.lines;
  }
}

export class UserView extends Container {
  constructor(text: string) {
    super();
    this.addChild(new Spacer(1));
    this.addChild(new Text(`${style.bold(style.cyan('›'))} ${text}`, 1, 0));
  }
}

export class AssistantView implements Component {
  public content: string = '';
  public thinking: string = '';
  public streaming: boolean = true;
  public errorMessage?: string;
  private readonly markdown = new Markdown('', 1, 0, markdownTheme);
  private cached?: { width: number; lines: string[] };

  constructor() {}

  public appendContent(delta: string) {
    this.content += delta;
    this.cached = undefined;
  }

  public appendThinking(delta: string) {
    this.thinking += delta;
    this.cached = undefined;
  }

  public finish(error?: string) {
    this.streaming = false;
    this.errorMessage = error;
    this.cached = undefined;
  }

  invalidate(): void {
    this.cached = undefined;
    this.markdown.invalidate();
  }

  render(width: number): string[] {
    if (this.cached?.width === width) return this.cached.lines;

    const lines: string[] = [];
    const reasoning = this.thinking.trim();
    if (reasoning) {
      const answering = this.content.trim().length > 0;
      if (this.streaming && !answering) {
        lines.push(style.dim(' thinking…'));
        const tail = wrapTextWithAnsi(reasoning.slice(-THINKING_TAIL_CHARS), Math.max(10, width - 4));
        for (const line of tail.slice(-THINKING_WINDOW_LINES)) lines.push(style.dim(`   ${line}`));
      } else {
        const words = reasoning.split(/\s+/).length;
        lines.push(style.dim(` thought for ${words} words`));
      }
    }

    const answer = this.content.trim();
    if (answer) {
      this.markdown.setText(answer);
      lines.push(...this.markdown.render(width));
    }

    if (this.errorMessage) {
      lines.push(style.red(` error: ${this.errorMessage}`));
    }

    if (lines.length > 0) lines.unshift('');
    this.cached = { width, lines };
    return lines;
  }
}

function plain(text: string): string {
  return stripTerminalSequences(text).replace(/\t/g, '    ').replace(/\r/g, '');
}

export class NoticeView implements Component {
  private text: string;

  constructor(text: string) {
    this.text = text;
  }

  invalidate(): void {}

  render(width: number): string[] {
    return [truncateToWidth(style.dim(` [${plain(this.text)}]`), width)];
  }
}

export class ToolView implements Component {
  public name: string;
  public summary: string;
  public state: 'running' | 'done' | 'error' = 'running';
  public resultText?: string;
  public diff?: string;
  private cached?: { width: number; lines: string[] };

  constructor(name: string, summary: string) {
    this.name = name;
    this.summary = summary;
  }

  public setResult(result: string, isError: boolean, diff?: string) {
    this.resultText = result;
    this.diff = diff;
    this.state = isError ? 'error' : 'done';
    this.cached = undefined;
  }

  invalidate(): void {
    this.cached = undefined;
  }

  render(width: number): string[] {
    if (this.cached?.width === width) return this.cached.lines;

    const glyph =
      this.state === 'running' ? style.yellow('…') : this.state === 'done' ? style.green('✓') : style.red('✗');
    const header = ` ${glyph} ${style.bold(this.name)} ${style.gray(plain(this.summary))}`;
    const lines = [truncateToWidth(header, width)];

    if (this.diff) {
      const diffLines = plain(this.diff).split('\n');
      const previewLines = diffLines.slice(0, DIFF_PREVIEW_LINES);
      for (const line of previewLines) {
        let styled: string;
        if (line.startsWith('+')) {
          styled = style.green(line);
        } else if (line.startsWith('-')) {
          styled = style.red(line);
        } else if (line.startsWith('@@')) {
          styled = style.cyan(line);
        } else if (line.startsWith('---') || line.startsWith('+++')) {
          styled = style.bold(line);
        } else {
          styled = style.gray(line);
        }
        lines.push(truncateToWidth(`   ${styled}`, width));
      }
      if (diffLines.length > DIFF_PREVIEW_LINES) {
        lines.push(
          truncateToWidth(
            style.dim(`   ... [${diffLines.length - DIFF_PREVIEW_LINES} more lines omitted]`),
            width
          )
        );
      }
    } else if (this.resultText && this.resultText !== '(command completed with no output)') {
      const maxLines = this.state === 'error' ? TOOL_PREVIEW_LINES : 4;
      const textLines = plain(this.resultText).trimEnd().split('\n').filter(Boolean).slice(0, maxLines);
      for (const line of textLines) {
        const styled = this.state === 'error' ? style.red(line) : style.dim(line);
        lines.push(truncateToWidth(`   ${styled}`, width));
      }
    }

    this.cached = { width, lines };
    return lines;
  }
}

export function formatTokens(count: number): string {
  return count < 1000 ? String(count) : `${(count / 1000).toFixed(1)}k`;
}

export function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  if (total < 60) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}m ${seconds}s`;
}

export interface BannerOptions {
  version: string;
  cwd: string;
  modelName?: string;
  mode?: string;
}

export class BannerView implements Component {
  public options: BannerOptions;
  private cached?: { width: number; lines: string[] };

  constructor(options: BannerOptions) {
    this.options = options;
  }

  public setModel(modelName?: string, mode?: string) {
    this.options.modelName = modelName;
    this.options.mode = mode;
    this.cached = undefined;
  }

  public setCwd(cwd: string) {
    this.options.cwd = cwd;
    this.cached = undefined;
  }

  invalidate(): void {
    this.cached = undefined;
  }

  render(width: number): string[] {
    if (this.cached?.width === width) return this.cached.lines;

    const boxWidth = Math.min(Math.max(40, width - 2), 62);
    const contentWidth = boxWidth - 4;

    const home = homedir();
    const displayCwd = this.options.cwd.startsWith(home)
      ? `~${this.options.cwd.slice(home.length)}`
      : this.options.cwd;

    const innerLines: string[] = [];

    // Title line
    const title = `${style.bold(style.cyan('⚡ DSH-Lite CLI'))} ${style.gray(`v${this.options.version}`)}`;
    innerLines.push(title);

    // Model line (ONLY if loaded by user!)
    if (this.options.modelName) {
      const modeTag = this.options.mode ? ` (${this.options.mode})` : '';
      const modelLine = `${style.dim('Model:     ')}${style.green(this.options.modelName)}${style.dim(modeTag)}`;
      innerLines.push(modelLine);
    }

    // Workspace line
    const cwdLine = `${style.dim('Workspace: ')}${style.yellow(displayCwd)}`;
    innerLines.push(cwdLine);

    // Hint line
    const hint = `${style.dim('Commands:  type ')}${style.cyan('/')}${style.dim(' for menu, ')}${style.cyan('/model')}${style.dim(' to select')}`;
    innerLines.push(hint);

    // Build the bordered box
    const top = style.cyan(`╭${'─'.repeat(boxWidth - 2)}╮`);
    const bottom = style.cyan(`╰${'─'.repeat(boxWidth - 2)}╯`);

    const lines: string[] = ['', top];
    for (const line of innerLines) {
      const truncated = truncateToWidth(line, contentWidth);
      const pad = ' '.repeat(Math.max(0, contentWidth - visibleWidth(truncated)));
      lines.push(`${style.cyan('│')}  ${truncated}${pad}${style.cyan('│')}`);
    }
    lines.push(bottom);

    this.cached = { width, lines };
    return lines;
  }
}

export interface ServeViewOptions {
  modelName: string;
  localUrl: string;
  remoteUrl: string;
  port: string;
}

export class ServeView implements Component {
  private options: ServeViewOptions;
  private logLines: string[] = [];
  private maxLogs = 100;
  private cached?: { width: number; lines: string[] };

  constructor(options: ServeViewOptions) {
    this.options = options;
  }

  public addLogLine(line: string) {
    this.logLines.push(line);
    if (this.logLines.length > this.maxLogs) {
      this.logLines.shift();
    }
    this.cached = undefined;
  }

  public invalidate(): void {
    this.cached = undefined;
  }

  public render(width: number): string[] {
    if (this.cached?.width === width) return this.cached.lines;

    const boxWidth = Math.max(20, Math.min(width, 90));
    const contentWidth = boxWidth - 4;

    const headerLines: string[] = [];
    headerLines.push(`${style.bold(style.cyan('🚀 Hosting Remote Model:'))} ${style.green(this.options.modelName)}`);
    headerLines.push(`${style.dim('Remote URL:')} ${style.bold(style.yellow(this.options.remoteUrl))}`);
    headerLines.push(`${style.dim('Local URL: ')} ${style.cyan(this.options.localUrl)}`);
    headerLines.push(`${style.dim('Control:   ')} Press ${style.bold(style.cyan('Esc'))} or ${style.bold(style.cyan('Ctrl+C'))} to stop server`);

    const top = style.cyan(`╭${'─'.repeat(boxWidth - 2)}╮`);
    const bottom = style.cyan(`╰${'─'.repeat(boxWidth - 2)}╯`);

    const lines: string[] = ['', top];
    for (const hLine of headerLines) {
      const truncated = truncateToWidth(hLine, contentWidth);
      const pad = ' '.repeat(Math.max(0, contentWidth - visibleWidth(truncated)));
      lines.push(`${style.cyan('│')}  ${truncated}${pad}${style.cyan('│')}`);
    }
    lines.push(bottom);
    lines.push('');
    lines.push(style.bold(style.dim('─── Live Server Logs ───────────────────────────────────────────')));

    if (this.logLines.length === 0) {
      lines.push(style.dim('  Waiting for llama-server output...'));
    } else {
      const displayLogs = this.logLines.slice(-15);
      for (const log of displayLogs) {
        if (log.includes('print_timing')) {
          lines.push(style.green(`  ${truncateToWidth(log, width - 4)}`));
        } else if (log.includes('error') || log.includes('ERR')) {
          lines.push(style.red(`  ${truncateToWidth(log, width - 4)}`));
        } else {
          lines.push(style.dim(`  ${truncateToWidth(log, width - 4)}`));
        }
      }
    }
    lines.push('');

    this.cached = { width, lines };
    return lines;
  }
}

export type AiStatus = 'idle' | 'thinking' | 'working' | 'serving';

export interface FooterState {
  modelName?: string;
  mode?: string;
  interactionMode?: string;
  aiStatus?: AiStatus;
  cwd: string;
  usedTokens?: number;
  contextWindow?: number;
  tokensPerSecond?: number;
  turnDurationMs?: number;
  isRunning?: boolean;
}

export function formatFooter(state: FooterState): string {
  const parts: string[] = [];

  if (state.modelName) {
    parts.push(style.cyan(state.modelName));

    const status: AiStatus =
      state.aiStatus ??
      (state.mode === 'thinking'
        ? 'thinking'
        : state.mode === 'working'
        ? 'working'
        : state.mode === 'serving'
        ? 'serving'
        : 'idle');

    let statusBracket: string;
    switch (status) {
      case 'thinking':
        statusBracket = style.yellow('[Thinking]');
        break;
      case 'working':
        statusBracket = style.cyan('[Working]');
        break;
      case 'serving':
        statusBracket = style.magenta('[Serving]');
        break;
      case 'idle':
      default:
        statusBracket = style.gray('[Idle]');
        break;
    }

    if (status === 'serving') {
      parts.push(statusBracket);
    } else if (state.interactionMode) {
      parts.push(`${style.yellow(state.interactionMode)} ${statusBracket}`);
    } else {
      parts.push(statusBracket);
    }

    if (state.contextWindow && state.contextWindow > 0) {
      const used = formatTokens(state.usedTokens ?? 0);
      const total = formatTokens(state.contextWindow);
      parts.push(style.gray(`ctx ${used}/${total}`));
    }
    if (state.tokensPerSecond && state.tokensPerSecond > 0) {
      parts.push(style.green(`${state.tokensPerSecond.toFixed(1)} tok/s`));
    }
  } else {
    parts.push(style.dim('no model loaded'));
    parts.push(style.cyan('/model to select'));
  }

  if (state.turnDurationMs !== undefined) {
    if (state.isRunning) {
      parts.push(style.yellow(formatDuration(state.turnDurationMs)));
    } else {
      parts.push(style.gray(`took ${formatDuration(state.turnDurationMs)}`));
    }
  }

  const home = homedir();
  const displayCwd = state.cwd.startsWith(home) ? `~${state.cwd.slice(home.length)}` : state.cwd;
  parts.push(style.gray(displayCwd));

  return style.gray(' ') + parts.join(style.gray(' · '));
}
