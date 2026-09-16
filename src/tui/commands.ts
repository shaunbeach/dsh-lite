import type { LiteModel } from '../config/models.js';

export type CommandName =
  | 'cd'
  | 'model'
  | 'mode'
  | 'agent'
  | 'plan'
  | 'chat'
  | 'serve'
  | 'disconnect'
  | 'clear'
  | 'resume'
  | 'quit';

export interface SlashCommand {
  name: string;
  description: string;
  argumentHint?: string;
  getArgumentCompletions?: (prefix: string) => { value: string; label: string }[];
}

export const COMMANDS: readonly { name: CommandName; description: string; argumentHint?: string }[] = [
  { name: 'agent', description: 'Switch to agent mode (autonomous coding with all tools)' },
  { name: 'plan', description: 'Switch to plan mode (design & implementation planning, read-only tools)' },
  { name: 'chat', description: 'Switch to chat mode (conversation plus web search, no file or shell tools)' },
  { name: 'cd', description: 'Change the workspace directory tools work in', argumentHint: '<path>' },
  { name: 'model', description: 'Switch model (picker lists local models)', argumentHint: '[name]' },
  { name: 'mode', description: 'Switch between thinking and instruct sampling', argumentHint: '[thinking|instruct]' },
  { name: 'serve', description: 'Serve a model as a remote host with live server logs' },
  { name: 'disconnect', description: 'Stop llama-server and unload model without exiting app' },
  { name: 'clear', description: 'Clear conversation and start a new session' },
  { name: 'resume', description: 'Resume a saved session', argumentHint: '[id]' },
  { name: 'quit', description: 'Exit dsh' },
];

const ALIASES: Record<string, CommandName> = { exit: 'quit', new: 'clear', stop: 'disconnect' };

export interface ParsedCommand {
  name: CommandName;
  args: string;
}

export function parseCommand(input: string): ParsedCommand | undefined {
  const match = /^\/([a-z]+)(?:\s+([\s\S]*))?$/.exec(input.trim());
  if (!match) return undefined;
  const name = ALIASES[match[1]] ?? COMMANDS.find((command) => command.name === match[1])?.name;
  return name ? { name, args: (match[2] ?? '').trim() } : undefined;
}

export function slashCommands(models: readonly LiteModel[]): SlashCommand[] {
  const complete = (values: readonly string[]) => (prefix: string) => {
    const lower = prefix.toLowerCase();
    const starting = values.filter((val) => val.toLowerCase().startsWith(lower));
    const matches = starting.length > 0 ? starting : values.filter((val) => val.toLowerCase().includes(lower));
    return matches.map((value) => ({ value, label: value }));
  };

  const argumentCompletions: Partial<Record<CommandName, (prefix: string) => { value: string; label: string }[]>> = {
    model: complete([...models.map((m) => m.name), 'deepseek-chat', 'deepseek-reasoner']),
    mode: complete(['thinking', 'instruct']),
  };

  return COMMANDS.map((cmd) => ({
    name: cmd.name,
    description: cmd.description,
    argumentHint: cmd.argumentHint,
    getArgumentCompletions: argumentCompletions[cmd.name],
  }));
}
