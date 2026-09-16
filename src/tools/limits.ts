const MIN_OUTPUT_BYTES = 4 * 1024;
const MAX_OUTPUT_BYTES = 50 * 1024;

/**
 * How many bytes of one tool result may reach the model, scaled to the context window: about a
 * quarter of it at roughly three bytes per token. A fixed cap either wastes a large window or lets
 * one command swallow a small one.
 */
export function toolOutputLimitBytes(contextWindow: number): number {
  return Math.min(MAX_OUTPUT_BYTES, Math.max(MIN_OUTPUT_BYTES, Math.floor(contextWindow * 0.25 * 3)));
}
