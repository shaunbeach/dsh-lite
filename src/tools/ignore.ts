/**
 * Directories the search tools skip. Kept in one place so `list_dir` and `grep_search` agree on
 * what the workspace contains. `.dsh` holds this agent's own session transcripts, which would
 * otherwise match every search for something the conversation discussed.
 */
export const IGNORE_DIRS: ReadonlySet<string> = new Set([
  '.git',
  '.dsh',
  'node_modules',
  'dist',
  '.next',
  '.venv',
  '__pycache__',
]);
