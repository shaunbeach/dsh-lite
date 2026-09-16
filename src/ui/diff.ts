import pc from 'picocolors';

export function renderDiff(diffString: string): string {
  if (!diffString) return '';

  const lines = diffString.split(/\r?\n/);
  const formattedLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith('---') || line.startsWith('+++')) {
      formattedLines.push(pc.bold(pc.gray(line)));
    } else if (line.startsWith('@@')) {
      formattedLines.push(pc.cyan(line));
    } else if (line.startsWith('+')) {
      formattedLines.push(pc.green(line));
    } else if (line.startsWith('-')) {
      formattedLines.push(pc.red(line));
    } else {
      formattedLines.push(pc.dim(line));
    }
  }

  return '\n' + formattedLines.join('\n') + '\n';
}
