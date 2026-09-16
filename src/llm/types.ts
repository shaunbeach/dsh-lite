export type Role = 'system' | 'user' | 'assistant' | 'tool';

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  role: Role;
  content: string | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
  reasoning_content?: string | null;
}

export interface StreamCallbacks {
  onReasoningDelta?: (delta: string) => void;
  onContentDelta?: (delta: string) => void;
  onToolCallDelta?: (index: number, id?: string, name?: string, argsDelta?: string) => void;
}

export interface CompletionResult {
  content: string;
  reasoningContent: string;
  toolCalls: ToolCall[];
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  predictedPerSecond?: number;
  durationMs?: number;
}
