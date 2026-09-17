import * as os from 'node:os';
import * as path from 'node:path';

/**
 * Shell commands that can destroy work irrecoverably, and are almost never what was meant.
 *
 * This is not a security boundary. A shell cannot be confined by inspecting its text, and anything
 * determined to get past this can. It exists because the model is not an adversary — it is a small
 * model acting on an instruction it may have misread, and "clear the workspace" is one plausible
 * reading away from deleting every file in it. The cost of being wrong is asymmetric: a refused
 * command is retyped, while deleted work is gone.
 *
 * The rule is narrow on purpose. Removing a build directory or a dependency tree is ordinary work
 * and stays allowed; what is refused is emptying the workspace itself, reaching outside it, or
 * throwing away uncommitted changes.
 */
export interface DestructiveVerdict {
  refused: boolean;
  reason?: string;
}

const HOME = os.homedir();

/** Splits a command on operators, so each part of `a && rm -rf b` is judged on its own. */
function segments(command: string): string[] {
  return command
    .split(/\s*(?:&&|\|\||;|\||\n)\s*/)
    .map(s => s.trim())
    .filter(Boolean);
}

/** Strips one layer of quoting from an argument, which is all a shell would do to a plain path. */
function unquote(token: string): string {
  const quoted = /^(['"])(.*)\1$/.exec(token);
  return quoted ? quoted[2] : token;
}

/**
 * Whether a deletion target is too broad to be an intended one.
 *
 * A glob standing for everything in a directory is judged by that directory, so `rm -rf .` and
 * `rm -rf ./*` are the same act and are refused alike.
 */
function targetVerdict(rawTarget: string, cwd: string): DestructiveVerdict {
  const target = unquote(rawTarget);
  const expanded = target.startsWith('~') ? path.join(HOME, target.slice(1)) : target;

  // A trailing glob deletes the contents of its parent, which is the thing being risked.
  const stripped = expanded.replace(/\/?\*+$/, '');
  const resolved = path.resolve(cwd, stripped || '.');

  if (resolved === path.parse(resolved).root) {
    return { refused: true, reason: 'that would delete the filesystem root' };
  }
  if (resolved === HOME) {
    return { refused: true, reason: 'that would delete the home directory' };
  }
  if (resolved === cwd) {
    return {
      refused: true,
      reason: `that would delete everything in the working directory (${cwd})`,
    };
  }
  if (!resolved.startsWith(cwd + path.sep)) {
    return { refused: true, reason: `${resolved} is outside the working directory` };
  }
  // .dsh holds the session transcripts, which are the only record of what a turn did and the only
  // way to recover something a turn wrote. "Clear the chat" read as an instruction lands here.
  if (resolved.split(path.sep).includes('.dsh')) {
    return { refused: true, reason: 'that would delete the session history in .dsh' };
  }
  return { refused: false };
}

/**
 * Judges one command. Returns a refusal with a reason, or a verdict that lets it run.
 */
export function judgeCommand(command: string, cwd: string): DestructiveVerdict {
  const workspace = path.resolve(cwd);

  for (const segment of segments(command)) {
    const tokens = segment.split(/\s+/).filter(Boolean);
    if (tokens.length === 0) continue;
    const [program, ...args] = tokens;

    if (program === 'rm') {
      const recursive = args.some(a => /^-[a-zA-Z]*[rR]/.test(a));
      const targets = args.filter(a => !a.startsWith('-'));
      // A non-recursive rm cannot empty a directory, so only the reach matters.
      for (const target of targets) {
        const verdict = targetVerdict(target, workspace);
        if (verdict.refused && (recursive || target.includes('*') || !verdict.reason?.includes('everything'))) {
          return verdict;
        }
      }
    }

    if (program === 'find' && args.includes('-delete')) {
      const root = args.find(a => !a.startsWith('-')) ?? '.';
      const verdict = targetVerdict(root, workspace);
      if (verdict.refused) return verdict;
    }

    if (program === 'git') {
      const joined = args.join(' ');
      if (/^reset\b.*--hard/.test(joined)) {
        return { refused: true, reason: 'git reset --hard discards uncommitted work' };
      }
      if (/^clean\b.*-[a-zA-Z]*[fd]/.test(joined)) {
        return { refused: true, reason: 'git clean deletes untracked files' };
      }
    }

    if (program === 'dd' || program === 'mkfs' || program === 'shred') {
      return { refused: true, reason: `${program} writes over data irrecoverably` };
    }
  }

  return { refused: false };
}

/** The message a refused command returns to the model, telling it what to do instead. */
export function refusalMessage(reason: string): string {
  return (
    `Refused: ${reason}. Name the specific files or directories to remove instead, ` +
    `or ask the user to do it themselves. If the intent was to start fresh, ` +
    `/clear resets the conversation and /project starts a new directory.`
  );
}
