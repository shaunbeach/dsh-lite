import OpenAI from 'openai';
import type { ChatMessage, CompletionResult, StreamCallbacks, ToolCall } from './types.js';
import type { SamplingConfig } from '../config/models.js';

export interface DeepSeekClientConfig {
  apiKey?: string;
  baseURL?: string;
  defaultModel?: string;
  samplingMode?: 'thinking' | 'instruct';
  /** Milliseconds of silence mid-request before the model is treated as stalled. */
  streamIdleTimeoutMs?: number;
}

/**
 * How long a request may produce nothing before it is abandoned. A large prompt can take minutes of
 * prompt processing before the first token, so this is generous; it exists to end the case where a
 * server never answers at all, which the 24-hour request timeout would otherwise hide until morning.
 */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000;

export const DEFAULT_SAMPLING: Record<'thinking' | 'instruct', SamplingConfig> = {
  thinking: {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 0.0,
    thinking: true,
  },
  instruct: {
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 1.5,
    thinking: false,
  },
};

/** A model's per-mode sampling from models.yml, kept whole so a `/mode` switch picks the right half. */
export interface ModelSampling {
  thinking?: SamplingConfig;
  instruct?: SamplingConfig;
}

export interface ModelConfiguration {
  model: string;
  /** Omitted keeps the current mode, for a switch that only changes the model. */
  mode?: 'thinking' | 'instruct';
  sampling?: ModelSampling;
  /** Whether the model's template has a thinking switch to send `enable_thinking` to. */
  reasoning?: boolean;
  /** llama.cpp accepts top_k, min_p and chat_template_kwargs; the DeepSeek cloud API rejects them. */
  extendedSampling?: boolean;
}

export class DeepSeekClient {
  private openai: OpenAI;
  public model: string;
  public mode: 'thinking' | 'instruct';
  public baseURL: string;
  public apiKey: string;
  public sampling?: ModelSampling;
  public reasoning = false;
  public extendedSampling = false;
  public streamIdleTimeoutMs: number;

  constructor(config: DeepSeekClientConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.DEEPSEEK_API_KEY ?? 'local-no-key';
    this.baseURL = config.baseURL ?? process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com';
    this.model = config.defaultModel ?? process.env.DEEPSEEK_MODEL ?? '';
    this.mode = config.samplingMode ?? 'thinking';
    const envIdle = Number(process.env.DSH_STREAM_IDLE_TIMEOUT_MS);
    this.streamIdleTimeoutMs =
      config.streamIdleTimeoutMs ??
      (Number.isFinite(envIdle) && envIdle > 0 ? envIdle : DEFAULT_STREAM_IDLE_TIMEOUT_MS);

    this.openai = new OpenAI({
      apiKey: this.apiKey,
      baseURL: this.baseURL,
      timeout: 86_400_000,
      maxRetries: 3,
    });
  }

  public setEndpoint(baseURL: string, apiKey = 'local-no-key') {
    this.baseURL = baseURL;
    this.apiKey = apiKey;
    this.openai = new OpenAI({
      apiKey,
      baseURL,
      timeout: 86_400_000,
      maxRetries: 3,
    });
  }

  /**
   * Points the client at a model. Every sampling field is replaced, not merged, so nothing from the
   * previous model survives a switch.
   */
  public configureModel(config: ModelConfiguration): void {
    this.model = config.model;
    if (config.mode) this.mode = config.mode;
    this.sampling = config.sampling;
    this.reasoning = config.reasoning ?? false;
    this.extendedSampling = config.extendedSampling ?? false;
  }

  /**
   * Sampling for the mode in effect right now. Resolved per request rather than stored, so `/mode`
   * selects the matching half of the model's config instead of leaving the other one applied.
   */
  public resolveSampling(): SamplingConfig {
    const base = DEFAULT_SAMPLING[this.mode];
    const override = this.sampling?.[this.mode];
    return {
      ...base,
      ...override,
      extra: { ...base.extra, ...override?.extra },
    };
  }

  public setMode(mode: 'thinking' | 'instruct') {
    this.mode = mode;
  }

  public hasModel(): boolean {
    return Boolean(this.model);
  }

  public async streamChat(
    messages: ChatMessage[],
    tools: OpenAI.Chat.ChatCompletionTool[] = [],
    callbacks: StreamCallbacks = {},
    abortSignal?: AbortSignal,
  ): Promise<CompletionResult> {
    if (!this.model) {
      throw new Error('No model has been loaded yet. Please select a model with /model.');
    }
    const formattedMessages = messages.map((m) => {
      const msg: OpenAI.Chat.ChatCompletionMessageParam = {
        role: m.role as any,
        content: m.content ?? '',
      };
      if (m.name) msg.name = m.name;
      if (m.tool_call_id) (msg as any).tool_call_id = m.tool_call_id;
      if (m.tool_calls && m.tool_calls.length > 0) {
        (msg as any).tool_calls = m.tool_calls;
      }
      return msg;
    });

    const sampling = this.resolveSampling();

    const requestBody: OpenAI.Chat.ChatCompletionCreateParamsStreaming & Record<string, unknown> = {
      model: this.model,
      messages: formattedMessages,
      tools: tools.length > 0 ? tools : undefined,
      stream: true,
      stream_options: { include_usage: true },
      temperature: sampling.temperature,
      top_p: sampling.top_p,
      presence_penalty: sampling.presence_penalty,
    };

    if (this.extendedSampling) {
      if (sampling.top_k !== undefined) requestBody.top_k = sampling.top_k;
      if (sampling.min_p !== undefined) requestBody.min_p = sampling.min_p;
      if (sampling.reasoningEffort !== undefined) {
        // llama.cpp templates define their own levels (xhigh), outside the SDK's union.
        requestBody.reasoning_effort = sampling.reasoningEffort as never;
      }
      // Only a reasoning template has a thinking switch; sending it to an instruct-only model
      // would be a template argument it cannot use.
      if (this.reasoning) {
        requestBody.chat_template_kwargs = {
          enable_thinking: sampling.thinking ?? true,
        };
      }
    }

    // `extra` wins over everything above, but chat_template_kwargs merges key by key so a model's
    // reasoning_effort does not erase the thinking switch.
    if (sampling.extra) {
      const { chat_template_kwargs: extraKwargs, ...rest } = sampling.extra;
      Object.assign(requestBody, rest);
      if (extraKwargs && typeof extraKwargs === 'object' && !Array.isArray(extraKwargs)) {
        requestBody.chat_template_kwargs = {
          ...(requestBody.chat_template_kwargs as Record<string, unknown> | undefined),
          ...extraKwargs,
        };
      }
    }

    const startTime = Date.now();
    let content = '';
    let reasoningContent = '';
    const toolCallsMap: Record<number, ToolCall> = {};
    let promptTokens = 0;
    let completionTokens = 0;
    let predictedPerSecond: number | undefined;

    // A stalled request must not wait for the 24-hour request timeout: abandon it once the server
    // has produced nothing for streamIdleTimeoutMs. The timer resets on every chunk, so a slow but
    // live generation runs as long as it needs.
    const idleController = new AbortController();
    let idleTimer: NodeJS.Timeout | undefined;
    let idleExpired = false;
    const resetIdleTimer = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        idleExpired = true;
        idleController.abort();
      }, this.streamIdleTimeoutMs);
    };
    const signal = abortSignal
      ? AbortSignal.any([abortSignal, idleController.signal])
      : idleController.signal;

    try {
      resetIdleTimer();
      const stream = await this.openai.chat.completions.create(requestBody, {
        signal,
      });

      for await (const chunk of stream) {
        resetIdleTimer();
        // llama-server chunk timings
        const rawChunk = chunk as any;
        if (rawChunk.timings) {
          if (rawChunk.timings.predicted_per_second) {
            predictedPerSecond = Number(rawChunk.timings.predicted_per_second);
          }
          if (rawChunk.timings.prompt_n !== undefined || rawChunk.timings.cache_n !== undefined) {
            promptTokens = Math.max(
              promptTokens,
              (rawChunk.timings.prompt_n ?? 0) + (rawChunk.timings.cache_n ?? 0),
            );
          }
          if (rawChunk.timings.predicted_n !== undefined) {
            completionTokens = Math.max(completionTokens, rawChunk.timings.predicted_n);
          }
        }

        if (rawChunk.usage) {
          if (rawChunk.usage.prompt_tokens) promptTokens = rawChunk.usage.prompt_tokens;
          if (rawChunk.usage.completion_tokens) completionTokens = rawChunk.usage.completion_tokens;
        }

        const choice = chunk.choices[0];
        if (!choice) continue;
        const delta = choice.delta as any;

        // Handle reasoning content (DeepSeek-R1 or llama.cpp reasoning delta)
        if (delta.reasoning_content) {
          const rDelta = delta.reasoning_content as string;
          reasoningContent += rDelta;
          callbacks.onReasoningDelta?.(rDelta);
        }

        // Handle standard text content
        if (delta.content) {
          const cDelta = delta.content as string;
          content += cDelta;
          callbacks.onContentDelta?.(cDelta);
        }

        // Handle streaming tool calls
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const index = tc.index ?? 0;
            if (!toolCallsMap[index]) {
              toolCallsMap[index] = {
                id: tc.id || '',
                type: 'function',
                function: {
                  name: tc.function?.name || '',
                  arguments: tc.function?.arguments || '',
                },
              };
            } else {
              if (tc.id) toolCallsMap[index].id = tc.id;
              if (tc.function?.name) toolCallsMap[index].function.name += tc.function.name;
              if (tc.function?.arguments) toolCallsMap[index].function.arguments += tc.function.arguments;
            }

            callbacks.onToolCallDelta?.(index, tc.id, tc.function?.name, tc.function?.arguments);
          }
        }
      }
    } catch (err) {
      if (idleExpired) {
        const seconds = Math.round(this.streamIdleTimeoutMs / 1000);
        throw new Error(
          `${this.model} stopped responding: no output for ${seconds}s. The server may still be busy with an earlier request — check it, or retry.`,
        );
      }
      throw err;
    } finally {
      clearTimeout(idleTimer);
    }

    const durationMs = Date.now() - startTime;
    if (completionTokens === 0 && (content || reasoningContent)) {
      completionTokens = Math.ceil((content.length + reasoningContent.length) / 3.8);
    }
    if (promptTokens === 0) {
      const totalChars = formattedMessages.reduce(
        (acc, m) => acc + (m.content ? String(m.content).length : 0),
        0,
      );
      promptTokens = Math.ceil(totalChars / 3.8) + formattedMessages.length * 4;
    }
    if (predictedPerSecond === undefined && completionTokens > 0 && durationMs > 100) {
      predictedPerSecond = Number((completionTokens / (durationMs / 1000)).toFixed(1));
    }

    const toolCalls = Object.keys(toolCallsMap)
      .sort((a, b) => Number(a) - Number(b))
      .map((k) => toolCallsMap[Number(k)]);

    return {
      content,
      reasoningContent,
      toolCalls,
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      predictedPerSecond,
      durationMs,
    };
  }
}
