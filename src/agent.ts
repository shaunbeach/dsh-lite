import type { ChatMessage } from './llm/types.js';
import { DeepSeekClient } from './llm/client.js';
import { ToolRegistry } from './tools/registry.js';
import { toolOutputLimitBytes } from './tools/limits.js';
import { SessionStore } from './session/store.js';
import { TokenStreamRenderer } from './ui/stream.js';
import { renderDiff } from './ui/diff.js';
import { format } from './ui/format.js';
import { buildSystemPrompt, type InteractionMode } from './prompt.js';
import { ContextManager, estimateTokens } from './context.js';
import { resolveHomePath } from './config/models.js';
import type { LiteModel } from './config/models.js';
import * as fs from 'node:fs';
import * as path from 'node:path';

export type { InteractionMode } from './prompt.js';

export interface AgentOptions {
  client: DeepSeekClient;
  cwd?: string;
  sessionId?: string;
  maxSteps?: number;
  contextWindow?: number;
  maxTokens?: number;
}

export interface TurnMetrics {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  predictedPerSecond?: number;
  stepDurationMs?: number;
  turnDurationMs: number;
}

export interface AgentTurnCallbacks {
  onReasoningDelta?: (delta: string) => void;
  onContentDelta?: (delta: string) => void;
  onAssistantStart?: () => void;
  onAssistantEnd?: () => void;
  onStepComplete?: (metrics: TurnMetrics) => void;
  onToolStart?: (toolCallId: string, name: string, argsSummary: string) => void;
  onToolEnd?: (
    toolCallId: string,
    name: string,
    execution: { result: string; isError: boolean; diff?: string }
  ) => void;
  onContextTrimmed?: (summary: string) => void;
  onNotice?: (message: string) => void;
  onCancelled?: () => void;
}

/**
 * Turns a configured step limit into the loop bound, with 0 meaning unlimited.
 *
 * A non-numeric limit must not reach the loop: `step < NaN` is false on the first comparison, so
 * the turn would end before calling the model and the agent would simply never answer.
 */
export function resolveMaxSteps(option: number | undefined, envValue?: string): number {
  const fromEnv = envValue && envValue.trim() ? Number(envValue) : undefined;
  const requested = option ?? fromEnv ?? 100;

  if (!Number.isInteger(requested) || requested < 0) {
    const source = option !== undefined ? 'maxSteps' : 'DSH_MAX_STEPS';
    throw new Error(`${source} must be a non-negative integer (0 for unlimited); got ${JSON.stringify(option ?? envValue)}.`);
  }

  return requested === 0 ? Infinity : requested;
}

/**
 * Tools each restricted mode exposes, by name. Modes absent from this table get everything
 * registered.
 *
 * Voice is absent deliberately. Spoken instructions are confirmed before they run — the daemon asks
 * before sending, and nothing reaches the model without an explicit yes — so the check on a
 * misheard command sits there rather than in a reduced tool set. A voice session that cannot run
 * the thing it just wrote cannot finish the work it was asked to do.
 */
const TOOLS_BY_MODE: Record<'plan' | 'chat', readonly string[]> = {
  plan: ['view_file', 'list_dir', 'grep_search', 'web_search', 'web_fetch'],
  chat: ['web_search', 'web_fetch'],
};

/** Stands in for a tool result the user aborted before the tool could run. */
export const CANCELLED_TOOL_RESULT = 'Tool call was not executed: the user aborted this turn.';

export class Agent {
  public client: DeepSeekClient;
  public cwd: string;
  public sessionId: string;
  public sessionStore: SessionStore;
  public registry: ToolRegistry;
  public messages: ChatMessage[] = [];
  public contextManager: ContextManager;
  public activeModel?: LiteModel;
  public lastTurnMetrics?: TurnMetrics;
  public interactionMode: InteractionMode = 'agent';
  public maxSteps: number;

  constructor(options: AgentOptions) {
    this.client = options.client;
    this.cwd = options.cwd || process.cwd();
    this.maxSteps = resolveMaxSteps(options.maxSteps, process.env.DSH_MAX_STEPS);
    this.sessionStore = new SessionStore(this.cwd);
    this.sessionId = options.sessionId || this.sessionStore.createSessionId();
    this.registry = new ToolRegistry();
    this.contextManager = new ContextManager(
      options.contextWindow || 32768,
      options.maxTokens || 4096
    );

    this.initSystemPrompt();
  }

  public setInteractionMode(mode: InteractionMode): void {
    this.interactionMode = mode;
    this.initSystemPrompt();
  }

  public setModel(model: LiteModel) {
    this.activeModel = model;
    this.client.setEndpoint(model.baseUrl);
    this.client.configureModel({
      model: model.name,
      mode: model.mode || (model.reasoning ? 'thinking' : 'instruct'),
      sampling: model.sampling,
      reasoning: model.reasoning,
      extendedSampling: true,
    });
    this.contextManager.contextWindow = model.contextWindow;
    this.contextManager.maxTokens = model.maxTokens;
  }

  /**
   * Moves the workspace that tools resolve paths against, without restarting anything.
   *
   * The conversation follows: it is copied into a session under the new workspace, because the
   * session log has to hold what the model is being sent. Leaving the old log behind would make the
   * two disagree the moment the next message landed.
   *
   * Returns the resolved absolute path.
   */
  public async setCwd(target: string): Promise<string> {
    const resolved = path.resolve(this.cwd, resolveHomePath(target.trim()));

    let stats: fs.Stats;
    try {
      stats = fs.statSync(resolved);
    } catch {
      throw new Error(`No such directory: ${resolved}`);
    }
    if (!stats.isDirectory()) throw new Error(`Not a directory: ${resolved}`);

    if (resolved === this.cwd) return resolved;

    const carried = this.messages.filter(m => m.role !== 'system');
    this.cwd = resolved;
    this.sessionStore = new SessionStore(resolved);
    this.sessionId = this.sessionStore.createSessionId();
    // The system prompt states the working directory, so it has to be rebuilt for the new one.
    this.initSystemPrompt();
    for (const message of carried) await this.sessionStore.appendMessage(this.sessionId, message);

    return resolved;
  }

  public isModelLoaded(): boolean {
    return Boolean(this.activeModel || this.client.hasModel());
  }

  /** Installs the system prompt for the current mode, keeping the conversation that follows it. */
  public initSystemPrompt() {
    const systemPrompt = buildSystemPrompt({
      cwd: this.cwd,
      platform: process.platform,
      interactionMode: this.interactionMode,
    });

    if (this.messages.length > 0 && this.messages[0].role === 'system') {
      this.messages[0].content = systemPrompt;
    } else {
      this.messages = [
        {
          role: 'system',
          content: systemPrompt,
        },
        ...this.messages.filter(m => m.role !== 'system'),
      ];
    }
  }

  /**
   * Replaces the conversation with a stored session and returns how many
   * messages came back.
   *
   * The system prompt is rebuilt rather than restored: it states the working
   * directory, platform and interaction mode in effect now, which need not be
   * the ones the session was recorded under. Session logs hold only user,
   * assistant and tool messages for that reason, so a resumed conversation
   * without this would reach the model with no instructions and no tool rules.
   */
  public async resume(sessionId: string): Promise<number> {
    const loaded = await this.sessionStore.loadSession(sessionId);
    this.sessionId = sessionId;
    this.messages = loaded.filter(m => m.role !== 'system');
    const restored = this.messages.length;
    this.initSystemPrompt();
    return restored;
  }

  /**
   * Starts an empty conversation in a new session.
   *
   * The messages are dropped explicitly. initSystemPrompt only rewrites the system message when one
   * is already present, which is what a mode switch needs, so calling it alone would leave the whole
   * previous conversation in place to be re-sent and re-processed on the next request.
   */
  public clearHistory(): void {
    this.sessionId = this.sessionStore.createSessionId();
    this.messages = [];
    this.lastTurnMetrics = undefined;
    this.initSystemPrompt();
  }

  private async recordToolResult(toolCallId: string, name: string, content: string): Promise<void> {
    const toolMsg: ChatMessage = {
      role: 'tool',
      name,
      tool_call_id: toolCallId,
      content,
    };
    this.messages.push(toolMsg);
    await this.sessionStore.appendMessage(this.sessionId, toolMsg);
  }

  public async runTurn(
    userInput: string,
    callbacksOrRenderer?: AgentTurnCallbacks | TokenStreamRenderer,
    abortSignal?: AbortSignal
  ): Promise<void> {
    const turnStartTime = Date.now();
    const userMsg: ChatMessage = {
      role: 'user',
      content: userInput,
    };
    this.messages.push(userMsg);
    await this.sessionStore.appendMessage(this.sessionId, userMsg);

    let callbacks: AgentTurnCallbacks;
    let isOneShotStream = false;

    if (callbacksOrRenderer instanceof TokenStreamRenderer) {
      isOneShotStream = true;
      callbacks = {
        onReasoningDelta: (delta) => callbacksOrRenderer.handleReasoningDelta(delta),
        onContentDelta: (delta) => callbacksOrRenderer.handleContentDelta(delta),
        onAssistantEnd: () => callbacksOrRenderer.finish(),
      };
    } else if (callbacksOrRenderer) {
      callbacks = callbacksOrRenderer;
    } else {
      isOneShotStream = true;
      const defaultRenderer = new TokenStreamRenderer();
      callbacks = {
        onReasoningDelta: (delta) => defaultRenderer.handleReasoningDelta(delta),
        onContentDelta: (delta) => defaultRenderer.handleContentDelta(delta),
        onAssistantEnd: () => defaultRenderer.finish(),
      };
    }

    const notifyCancelled = () => {
      if (callbacks.onCancelled) {
        callbacks.onCancelled();
      } else if (isOneShotStream) {
        process.stdout.write(format.dim('\nOperation cancelled by user.\n'));
      }
    };

    let step = 0;

    while (step < this.maxSteps) {
      if (abortSignal?.aborted) {
        notifyCancelled();
        break;
      }

      step++;
      const stepStartTime = Date.now();

      const allowed =
        this.interactionMode === 'plan' || this.interactionMode === 'chat'
          ? TOOLS_BY_MODE[this.interactionMode]
          : undefined;
      const tools = allowed
        ? this.registry.getOpenAITools().filter(t => allowed.includes(t.function.name))
        : this.registry.getOpenAITools();

      // The tool schemas travel with every request and are rendered into the
      // prompt, so they come out of the same window the messages do.
      const toolTokens = tools.length > 0 ? estimateTokens(JSON.stringify(tools)) : 0;

      // Compact messages if context exceeds budget
      const contextSelection = this.contextManager.selectContext(this.messages, toolTokens);
      if (contextSelection.trimmed && contextSelection.summary) {
        if (callbacks.onContextTrimmed) {
          callbacks.onContextTrimmed(contextSelection.summary);
        } else if (isOneShotStream) {
          process.stdout.write(format.dim(`[context trimmed: ${contextSelection.summary}]\n`));
        }
      }

      if (!contextSelection.fits) {
        const model = this.activeModel?.name || this.client.model;
        throw new Error(
          `This turn does not fit ${model}'s ${this.contextManager.contextWindow}-token context window, ` +
            `even after compaction. Start a new session with /clear, or switch to a model with a larger window.`
        );
      }

      callbacks.onAssistantStart?.();

      let completion;
      try {
        completion = await this.client.streamChat(
          contextSelection.messages,
          tools,
          {
            onReasoningDelta: (delta) => callbacks.onReasoningDelta?.(delta),
            onContentDelta: (delta) => callbacks.onContentDelta?.(delta),
          },
          abortSignal
        );
      } catch (err: any) {
        callbacks.onAssistantEnd?.();
        if (abortSignal?.aborted || err.name === 'AbortError' || err.name === 'APIUserAbortError') {
          notifyCancelled();
          break;
        }
        throw err;
      }

      callbacks.onAssistantEnd?.();

      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: completion.content || null,
        tool_calls: completion.toolCalls.length > 0 ? completion.toolCalls : undefined,
        reasoning_content: completion.reasoningContent || undefined,
      };

      this.messages.push(assistantMsg);
      await this.sessionStore.appendMessage(this.sessionId, assistantMsg);

      // Record metrics
      const stepDurationMs = Date.now() - stepStartTime;
      const turnDurationMs = Date.now() - turnStartTime;
      this.lastTurnMetrics = {
        promptTokens: completion.promptTokens ?? 0,
        completionTokens: completion.completionTokens ?? 0,
        totalTokens: (completion.promptTokens ?? 0) + (completion.completionTokens ?? 0),
        predictedPerSecond: completion.predictedPerSecond,
        stepDurationMs,
        turnDurationMs,
      };

      callbacks.onStepComplete?.(this.lastTurnMetrics);

      // If no tool calls were requested, the assistant is done
      if (completion.toolCalls.length === 0) {
        // A turn that ends without an answer renders as nothing, which reads as the app freezing.
        // Reasoning does not count: it is shown collapsed as "thought for N words", so a model that
        // thinks and then writes no content leaves the user with a blank screen either way.
        const answered = (completion.content ?? '').trim().length > 0;
        if (!answered) {
          const model = this.activeModel?.name || this.client.model;
          const thoughtOnly = (completion.reasoningContent ?? '').trim().length > 0;
          const notice = thoughtOnly
            ? `${model} finished thinking but wrote no answer. Reasoning models sometimes end the turn inside the thinking block; ask again, or switch with /mode instruct.`
            : `${model} returned an empty response. The prompt may exceed what the server was launched with, or the chat template may have rejected the conversation.`;
          if (callbacks.onNotice) {
            callbacks.onNotice(notice);
          } else if (isOneShotStream) {
            process.stdout.write(format.dim(`\n[${notice}]\n`));
          }
        }
        break;
      }

      // Execute all tool calls. Every tool_call id must be answered by a tool
      // message or the next request is malformed, so an abort still records a
      // result for each call: the real one where the tool already ran, a
      // cancellation notice for the rest.
      let cancelled = false;

      for (const toolCall of completion.toolCalls) {
        const name = toolCall.function.name;

        if (cancelled || abortSignal?.aborted) {
          cancelled = true;
          await this.recordToolResult(toolCall.id, name, CANCELLED_TOOL_RESULT);
          continue;
        }

        const argsStr = toolCall.function.arguments;

        let argsSummary = '';
        try {
          const parsed = JSON.parse(argsStr);
          if (parsed.command) argsSummary = `"${parsed.command}"`;
          else if (parsed.path) argsSummary = parsed.path;
          else if (parsed.query) argsSummary = `"${parsed.query}"`;
        } catch {
          // ignore
        }

        if (callbacks.onToolStart) {
          callbacks.onToolStart(toolCall.id, name, argsSummary);
        } else if (isOneShotStream) {
          process.stdout.write(format.toolStart(name, argsSummary));
        }

        const execution = await this.registry.execute(
          name,
          argsStr,
          toolCall.id,
          {
            cwd: this.cwd,
            abortSignal,
            outputLimitBytes: toolOutputLimitBytes(this.contextManager.contextWindow),
          }
        );

        if (callbacks.onToolEnd) {
          callbacks.onToolEnd(toolCall.id, name, execution);
        } else if (isOneShotStream) {
          process.stdout.write(format.toolDone(name, execution.isError) + '\n');
          if (execution.diff) {
            process.stdout.write(renderDiff(execution.diff));
          }
        }

        // The tool ran, so its result is recorded even when the abort landed
        // mid-execution: the side effects already happened.
        await this.recordToolResult(toolCall.id, name, execution.result);

        if (abortSignal?.aborted) cancelled = true;
      }

      if (cancelled) {
        notifyCancelled();
        break;
      }

      // If we reached max steps and more tool steps remain
      if (step >= this.maxSteps) {
        const notice = `Reached turn step limit (${this.maxSteps} steps). Type "continue" to proceed or set DSH_MAX_STEPS=0 for unlimited.`;
        if (callbacks.onNotice) {
          callbacks.onNotice(notice);
        } else if (isOneShotStream) {
          process.stdout.write(format.dim(`\n[${notice}]\n`));
        }
      }
    }
  }
}
