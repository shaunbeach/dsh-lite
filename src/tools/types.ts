import type OpenAI from 'openai';

export interface ToolExecutionContext {
  cwd: string;
  onProgress?: (message: string) => void;
  abortSignal?: AbortSignal;
  /** Bytes of this tool's output that may reach the model, scaled to the model's context window. */
  outputLimitBytes?: number;
}

export interface ToolDefinition<TArgs = any, TResult = any> {
  name: string;
  description: string;
  parameters: OpenAI.Chat.ChatCompletionTool['function']['parameters'];
  execute: (args: TArgs, context: ToolExecutionContext) => Promise<TResult>;
}

export interface ToolExecutionResult {
  toolCallId: string;
  name: string;
  result: string;
  isError: boolean;
  diff?: string; // Optional diff string for edit operations
}
