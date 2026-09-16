import { type ChildProcess, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

const SHELL = existsSync('/bin/bash') ? '/bin/bash' : 'sh';

/** Process groups of commands still running, so process exit can kill them all. */
const runningGroups = new Set<number>();

function killGroup(pid: number): void {
  try {
    process.kill(-pid, 'SIGKILL');
  } catch {
    try {
      process.kill(pid, 'SIGKILL');
    } catch (err) {
      void err; // Already gone.
    }
  }
}

/** Kills every command still running. Detached groups would otherwise outlive the harness. */
export function killRunningCommands(): void {
  for (const pid of runningGroups) killGroup(pid);
  runningGroups.clear();
}

export interface ShellRunOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  onData: (data: Buffer) => void;
}

export interface ShellRunResult {
  /** null when the process was killed by a signal. */
  exitCode: number | null;
  timedOut: boolean;
}

/**
 * Runs a command under bash in its own process group, so an abort or timeout kills everything it
 * started rather than only the shell. Stdin is closed and pagers are disabled, so a command cannot
 * block on input that never arrives.
 */
export async function runShellCommand(
  command: string,
  cwd: string,
  options: ShellRunOptions
): Promise<ShellRunResult> {
  const { signal, timeoutMs, onData } = options;
  signal?.throwIfAborted();

  const child = spawn(SHELL, ['-c', command], {
    cwd,
    detached: true,
    env: { ...process.env, PAGER: 'cat', GIT_PAGER: 'cat' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const pid = child.pid;
  if (pid !== undefined) runningGroups.add(pid);

  let timedOut = false;
  const kill = () => {
    if (pid !== undefined) killGroup(pid);
  };

  const timer = timeoutMs === undefined ? undefined : setTimeout(() => {
    timedOut = true;
    kill();
  }, timeoutMs);

  const onAbort = () => kill();
  signal?.addEventListener('abort', onAbort, { once: true });

  try {
    const exitCode = await waitForChild(child, onData);
    return { exitCode, timedOut };
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
    if (pid !== undefined) runningGroups.delete(pid);
  }
}

/** How long to keep reading output after exit while a detached descendant still holds the pipes. */
const EXIT_STDIO_GRACE_MS = 100;

/**
 * Waits for the shell to exit without hanging on pipes a backgrounded descendant inherited.
 * `close` fires only once every copy of the pipe is shut, so a command that starts a server and
 * returns would otherwise block until that server died. Settling on `exit` plus a short quiet
 * period keeps the output while letting the command return.
 */
function waitForChild(child: ChildProcess, onData: (data: Buffer) => void): Promise<number | null> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let exited = false;
    let exitCode: number | null = null;
    let graceTimer: NodeJS.Timeout | undefined;

    const finalize = (code: number | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(graceTimer);
      child.stdout?.destroy();
      child.stderr?.destroy();
      resolve(code);
    };

    const armGrace = () => {
      clearTimeout(graceTimer);
      graceTimer = setTimeout(() => finalize(exitCode), EXIT_STDIO_GRACE_MS);
    };

    const handleData = (data: Buffer) => {
      onData(data);
      if (exited) armGrace();
    };

    child.stdout?.on('data', handleData);
    child.stderr?.on('data', handleData);

    child.once('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(graceTimer);
      reject(err);
    });
    child.once('exit', (code) => {
      exited = true;
      exitCode = code;
      armGrace();
    });
    child.once('close', (code) => finalize(code ?? exitCode));
  });
}
