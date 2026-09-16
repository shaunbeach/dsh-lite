import type { ChatMessage } from './llm/types.js';

export const TRIM_TARGET = 0.6; // Trim down to 60% of budget in one go
export const RECENT_STEPS_TO_PRESERVE = 2;

/** Fast heuristic token estimator (~4 chars per token for code/text). */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 3.8);
}

/**
 * Tokens one message costs in a request. `reasoning_content` is deliberately not counted: the client
 * does not forward it, so charging the budget for it would trim real history to make room for text
 * that never leaves this process.
 */
export function estimateMessageTokens(message: ChatMessage): number {
  let count = 4; // overhead per message
  if (message.content) count += estimateTokens(message.content);
  if (message.tool_calls) {
    for (const tc of message.tool_calls) {
      count += estimateTokens(tc.function.name) + estimateTokens(tc.function.arguments) + 8;
    }
  }
  return count;
}

export function elideToolResultContent(content: string, maxLines = 15): string {
  const lines = content.split('\n');
  if (lines.length <= maxLines) return content;
  const head = lines.slice(0, 5).join('\n');
  const tail = lines.slice(-5).join('\n');
  const elidedCount = lines.length - 10;
  return `${head}\n[elided from context: ${elidedCount} lines]\n${tail}`;
}

/**
 * Cuts `content` down to roughly `maxTokens`, keeping the start and the end.
 * The middle is where a file listing or command output is least informative.
 */
export function truncateToTokens(content: string, maxTokens: number): string {
  const marker = '\n[truncated to fit the context window]\n';
  const budget = Math.floor(maxTokens * 3.8) - marker.length;
  if (budget <= 0) return marker.trim();
  if (content.length <= budget) return content;
  const head = Math.ceil(budget * 0.6);
  const tail = budget - head;
  return content.slice(0, head) + marker + (tail > 0 ? content.slice(-tail) : '');
}

/** A user message and every assistant and tool message that answers it. */
interface Turn {
  messages: ChatMessage[];
}

/**
 * Groups messages into turns so compaction drops them whole. Dropping a
 * partial turn would separate a tool result from the assistant message that
 * requested it, which no chat template can render.
 */
function splitTurns(messages: ChatMessage[]): Turn[] {
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === 'user' || turns.length === 0) {
      turns.push({ messages: [message] });
    } else {
      turns[turns.length - 1].messages.push(message);
    }
  }
  return turns;
}

/**
 * Removes tool messages whose requesting assistant message is absent, which a
 * resumed session log can carry in before compaction ever runs.
 */
function dropOrphanToolMessages(messages: ChatMessage[]): { messages: ChatMessage[]; dropped: number } {
  const announced = new Set<string>();
  const kept: ChatMessage[] = [];
  let dropped = 0;

  for (const message of messages) {
    for (const call of message.tool_calls ?? []) announced.add(call.id);
    if (message.role === 'tool' && message.tool_call_id && !announced.has(message.tool_call_id)) {
      dropped++;
      continue;
    }
    kept.push(message);
  }

  return { messages: kept, dropped };
}

/** The retained message whose content is worth truncating first. */
function largestByContent(messages: ChatMessage[]): ChatMessage | undefined {
  let largest: ChatMessage | undefined;
  for (const message of messages) {
    if (!message.content) continue;
    if (!largest || message.content.length > largest.content!.length) largest = message;
  }
  return largest;
}

export interface ContextSelection {
  messages: ChatMessage[];
  trimmed: boolean;
  summary?: string;
  /** False when the newest turn exceeds the budget even after truncation. */
  fits: boolean;
}

export class ContextManager {
  public contextWindow: number;
  public maxTokens: number;

  constructor(contextWindow = 32768, maxTokens = 4096) {
    this.contextWindow = contextWindow;
    this.maxTokens = maxTokens;
  }

  public get budgetTokens(): number {
    return Math.max(1024, this.contextWindow - this.maxTokens);
  }

  /**
   * Fits the conversation into the model's context window.
   *
   * Compaction escalates: old tool output, then whole old turns, then tool
   * output in the turns that remain, and finally a hard truncation of whatever
   * is still oversized. The newest turn is never
   * dropped, and the result never contains a tool message without the
   * assistant message that requested it.
   *
   * `reserveTokens` covers what the request carries besides messages, such as
   * the tool schemas. A false `fits` means the caller must not send: the
   * request would exceed the window no matter what the server does with it.
   */
  public selectContext(messages: ChatMessage[], reserveTokens = 0): ContextSelection {
    const total = (list: ChatMessage[]) => list.reduce((acc, m) => acc + estimateMessageTokens(m), 0);
    const budget = Math.max(1024, this.budgetTokens - reserveTokens);

    // An orphaned tool result breaks the request at any size, and a resumed
    // log can carry one into a conversation that is nowhere near the budget.
    const sanitized = dropOrphanToolMessages(messages);
    let orphansDropped = sanitized.dropped;

    if (total(sanitized.messages) <= budget) {
      return {
        messages: sanitized.messages,
        trimmed: orphansDropped > 0,
        summary: orphansDropped > 0 ? `${orphansDropped} unanswerable tool results dropped` : undefined,
        fits: true,
      };
    }

    const target = budget * TRIM_TARGET;
    const result: ChatMessage[] = sanitized.messages.map(m => ({ ...m }));
    let elidedOutputs = 0;
    let droppedTurns = 0;
    let truncatedOutputs = 0;

    // 1. Elide verbose tool results from all but the most recent steps.
    if (total(result) > target) {
      let recentAssistantIdx = -1;
      let seen = 0;
      for (let i = result.length - 1; i >= 0; i--) {
        if (result[i].role === 'assistant' && ++seen === RECENT_STEPS_TO_PRESERVE) {
          recentAssistantIdx = i;
          break;
        }
      }

      if (recentAssistantIdx > 0) {
        for (let i = 0; i < recentAssistantIdx; i++) {
          if (result[i].role === 'tool' && result[i].content) {
            const before = result[i].content!;
            const elided = elideToolResultContent(before);
            if (elided !== before) {
              result[i].content = elided;
              elidedOutputs++;
            }
          }
        }
      }
    }

    const systemMessage = result.length > 0 && result[0].role === 'system' ? result.shift() : undefined;
    const systemTokens = systemMessage ? estimateMessageTokens(systemMessage) : 0;

    // 2. Drop the oldest turns whole. The newest turn holds the question being
    //    answered, so it always survives this step.
    let turns = splitTurns(result);
    while (turns.length > 1 && systemTokens + total(turns.flatMap(t => t.messages)) > target) {
      turns = turns.slice(1);
      droppedTurns++;
    }

    let kept = turns.flatMap(t => t.messages);

    // 3. Elide tool output in the turns that remain. Step 1 spares the recent
    //    steps to keep them readable, but fitting the window comes first.
    if (systemTokens + total(kept) > budget) {
      for (const message of kept) {
        if (message.role === 'tool' && message.content) {
          const before = message.content;
          const elided = elideToolResultContent(before);
          if (elided !== before) {
            message.content = elided;
            elidedOutputs++;
          }
        }
        if (systemTokens + total(kept) <= budget) break;
      }
    }

    // 4. Truncate what is still oversized, largest first. One tool result or
    //    pasted file can exceed the window on its own, and elision cannot help
    //    when it has no line breaks to cut on.
    while (systemTokens + total(kept) > budget) {
      const largest = largestByContent(kept);
      if (!largest) break;
      const excess = systemTokens + total(kept) - budget;
      const allowance = Math.max(64, estimateTokens(largest.content!) - excess - 16);
      const before = largest.content!;
      largest.content = truncateToTokens(before, allowance);
      if (largest.content === before) break;
      truncatedOutputs++;
    }

    const orphans = dropOrphanToolMessages(kept);
    kept = orphans.messages;
    orphansDropped += orphans.dropped;

    const finalMessages = systemMessage ? [systemMessage, ...kept] : kept;
    const trimmed =
      elidedOutputs > 0 || droppedTurns > 0 || truncatedOutputs > 0 || orphansDropped > 0;

    const summaryParts: string[] = [];
    if (elidedOutputs > 0) summaryParts.push(`${elidedOutputs} tool outputs elided`);
    if (droppedTurns > 0) summaryParts.push(`${droppedTurns} old turns dropped`);
    if (truncatedOutputs > 0) summaryParts.push(`${truncatedOutputs} oversized messages truncated`);
    if (orphansDropped > 0) summaryParts.push(`${orphansDropped} unanswerable tool results dropped`);

    return {
      messages: finalMessages,
      trimmed,
      summary: trimmed ? summaryParts.join(', ') : undefined,
      fits: total(finalMessages) <= budget,
    };
  }
}
