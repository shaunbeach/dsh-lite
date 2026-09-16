import pc from 'picocolors';

export const format = {
  reasoningHeader(): string {
    return pc.dim('\n╭─ Thinking (DeepSeek-R1) ───────────────────────────────────╮\n');
  },

  reasoningFooter(): string {
    return pc.dim('╰────────────────────────────────────────────────────────────╯\n\n');
  },

  toolStart(name: string, argsSummary?: string): string {
    const icon = getToolIcon(name);
    const label = pc.bold(pc.blue(`[${name}]`));
    const args = argsSummary ? pc.dim(` ${argsSummary}`) : '';
    return `${icon} ${label}${args}...`;
  },

  toolDone(_name: string, isError: boolean = false): string {
    const status = isError ? pc.red('✗ Failed') : pc.green('✓ Done');
    return ` ${status}`;
  },

  error(msg: string): string {
    return pc.bold(pc.red(`Error: `)) + msg;
  },

  info(msg: string): string {
    return pc.cyan(`ℹ `) + msg;
  },

  success(msg: string): string {
    return pc.green(`✓ `) + msg;
  },

  dim(msg: string): string {
    return pc.dim(msg);
  },

};

function getToolIcon(name: string): string {
  switch (name) {
    case 'bash':
      return pc.yellow('💻');
    case 'view_file':
      return pc.cyan('📖');
    case 'write_file':
      return pc.green('📝');
    case 'edit_file':
      return pc.magenta('✏️ ');
    case 'list_dir':
      return pc.blue('📁');
    case 'grep_search':
      return pc.cyan('🔍');
    default:
      return pc.gray('⚙️ ');
  }
}
