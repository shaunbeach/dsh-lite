/** Milliseconds a network tool waits before giving up, overridable for slow links. */
export function networkTimeoutMs(envValue: string | undefined, fallback: number): number {
  const parsed = Number(envValue);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}
