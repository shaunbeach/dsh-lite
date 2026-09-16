import { DeepSeekClient } from './llm/client.js';
import { Agent } from './agent.js';
import { TokenStreamRenderer } from './ui/stream.js';
import { format } from './ui/format.js';
import { loadModelsConfig, type LiteModel } from './config/models.js';
import { LlamaServerManager, stopServerOnExit } from './llm/server.js';
import pc from 'picocolors';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const VERSION = '0.2.0';

/**
 * Whether the compiled output this process is running predates the sources beside it.
 * `bin/dsh.js` runs `dist/`, so editing `src/` changes nothing until a build runs, and the old
 * behaviour is indistinguishable from the change not working. Absent sources mean an installed
 * package, where there is nothing to compare.
 */
function staleBuildWarning(): string | undefined {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(here, '..');
  const srcDir = path.join(root, 'src');
  const builtEntry = path.join(root, 'dist', 'cli.js');
  if (!fs.existsSync(srcDir) || !fs.existsSync(builtEntry)) return undefined;

  const builtAt = fs.statSync(builtEntry).mtimeMs;
  let newestSource = 0;
  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith('.ts')) newestSource = Math.max(newestSource, fs.statSync(full).mtimeMs);
    }
  };
  walk(srcDir);

  return newestSource > builtAt
    ? 'Running a stale build: src/ has changed since dist/ was compiled. Run `npm run build` to pick the changes up.'
    : undefined;
}

interface CliArgs {
  task?: string;
  model?: string;
  mode?: 'thinking' | 'instruct';
  modelsPath?: string;
  listModels: boolean;
  resumeSessionId?: string;
  maxSteps?: number;
  cwd?: string;
  help: boolean;
  version: boolean;
}

function parseArgs(args: string[]): CliArgs {
  const result: CliArgs = {
    help: false,
    version: false,
    listModels: false,
  };

  const positional: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--help' || arg === '-h') {
      result.help = true;
    } else if (arg === '--version' || arg === '-v') {
      result.version = true;
    } else if (arg === '--list-models') {
      result.listModels = true;
    } else if (arg === '--max-steps') {
      const raw = args[++i];
      const parsed = raw === undefined ? Number.NaN : Number(raw);
      if (!Number.isInteger(parsed) || parsed < 0) {
        throw new Error(`--max-steps needs a non-negative integer (0 for unlimited); got ${raw ?? '(no value)'}`);
      }
      result.maxSteps = parsed;
    } else if (arg === '--model' || arg === '-m') {
      result.model = args[++i];
    } else if (arg === '--mode') {
      const raw = args[++i];
      if (raw !== 'thinking' && raw !== 'instruct') {
        throw new Error(`--mode must be "thinking" or "instruct"; got ${raw ?? '(no value)'}`);
      }
      result.mode = raw;
    } else if (arg === '--models') {
      result.modelsPath = args[++i];
    } else if (arg === '--cwd') {
      result.cwd = args[++i];
    } else if (arg === '--resume' || arg === '-r') {
      const next = args[i + 1];
      if (next && !next.startsWith('-')) {
        result.resumeSessionId = next;
        i++;
      } else {
        result.resumeSessionId = 'LATEST';
      }
    } else if (!arg.startsWith('-')) {
      positional.push(arg);
    }
  }

  if (positional.length > 0) {
    result.task = positional.join(' ');
  }

  return result;
}

function printCliHelp() {
  console.log(`
${pc.bold('dsh-lite')} - Lean terminal coding harness for local llama.cpp models & DeepSeek

${pc.bold('USAGE:')}
  dsh [options] [task...]

${pc.bold('ARGUMENTS:')}
  [task...]                  Optional one-shot task to execute and exit.
                             If omitted, enters the interactive Claude Code-style REPL.

${pc.bold('OPTIONS:')}
  -m, --model <model>        Model name from models.yml (or deepseek-chat / deepseek-reasoner for cloud)
  --mode <thinking|instruct> Initial sampling mode (thinking or instruct)
  --models <path>            Path to models.yml configuration file
  --list-models              List available models in models.yml and exit
  --max-steps <n>            Maximum tool steps per turn (default: 100, 0 for unlimited)
  -r, --resume [id]          Resume a previous session (or latest session if id is omitted)
  --cwd <path>               Target workspace directory (default: current directory)
  -h, --help                 Show this help message
  -v, --version              Show version number

${pc.bold('EXAMPLES:')}
  dsh                        Start interactive REPL with default local model (or cloud)
  dsh -m Qwen3-4B            Start REPL with a specific local GGUF model from models.yml
  dsh --mode instruct        Start in instruct mode (no thinking tokens)
  dsh "run npm test"         Run a single task directly in the terminal and exit
  dsh --resume               Resume the most recent conversation session
`);
}

export async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.version) {
    console.log(`dsh-lite v${VERSION}`);
    process.exit(0);
  }

  if (args.help) {
    printCliHelp();
    process.exit(0);
  }

  // Load models.yml configuration if present
  const modelsConfig = loadModelsConfig(args.modelsPath);

  if (args.listModels) {
    if (!modelsConfig || modelsConfig.models.length === 0) {
      console.log('No local models configured. Create models.yml or pass --models <path>.');
    } else {
      console.log(pc.bold(`Models from ${modelsConfig.configPath}:`));
      for (const m of modelsConfig.models) {
        const reasoningTag = m.reasoning ? pc.magenta('[reasoning]') : pc.blue('[instruct]');
        console.log(`  ${pc.cyan(m.name)} ${reasoningTag} (ctx: ${m.contextWindow}, maxTokens: ${m.maxTokens})`);
        console.log(`    GGUF: ${pc.dim(m.modelPath)}`);
      }
    }
    process.exit(0);
  }

  // Determine target model (only if explicitly specified by user via --model or -m)
  let targetLocalModel: LiteModel | undefined;
  if (modelsConfig && modelsConfig.models.length > 0 && args.model) {
    targetLocalModel = modelsConfig.models.find(
      (m) => m.name.toLowerCase() === args.model!.toLowerCase() || m.name.toLowerCase().includes(args.model!.toLowerCase())
    );
  }

  const client = new DeepSeekClient({
    defaultModel: targetLocalModel ? targetLocalModel.name : args.model,
    baseURL: targetLocalModel ? targetLocalModel.baseUrl : undefined,
    samplingMode: args.mode || (targetLocalModel?.reasoning ? 'thinking' : 'instruct'),
  });

  const agent = new Agent({
    client,
    cwd: args.cwd,
    contextWindow: targetLocalModel?.contextWindow,
    maxTokens: targetLocalModel?.maxTokens,
    maxSteps: args.maxSteps,
  });

  if (targetLocalModel) {
    agent.setModel(targetLocalModel);
  }

  // Handle session resume
  if (args.resumeSessionId) {
    if (args.resumeSessionId === 'LATEST') {
      const sessions = await agent.sessionStore.listSessions();
      if (sessions.length > 0) {
        await agent.resume(sessions[0].id);
        console.log(format.info(`Resumed latest session: ${pc.cyan(sessions[0].id)}`));
      } else {
        console.log(format.info('No previous sessions found. Starting fresh session.'));
      }
    } else {
      try {
        await agent.resume(args.resumeSessionId);
        console.log(format.info(`Resumed session: ${pc.cyan(args.resumeSessionId)}`));
      } catch (err: any) {
        console.error(format.error(`Could not resume session ${args.resumeSessionId}: ${err.message}`));
        process.exit(1);
      }
    }
  }

  // If local model is selected and we are in one-shot mode, ensure llama-server is ready
  if (args.task) {
    if (!targetLocalModel && !agent.isModelLoaded() && modelsConfig && modelsConfig.models.length > 0) {
      targetLocalModel = modelsConfig.models[0];
      agent.setModel(targetLocalModel);
    }
    let serverManager: LlamaServerManager | undefined;
    if (targetLocalModel) {
      serverManager = new LlamaServerManager();
      stopServerOnExit(serverManager);
      await serverManager.ensure(targetLocalModel, (msg) => console.log(format.dim(`  ${msg}`)));
    }

    console.log(pc.bold(pc.cyan(`⚡ Running task: `)) + args.task);
    const renderer = new TokenStreamRenderer();
    try {
      await agent.runTurn(args.task, renderer);
    } catch (err: any) {
      console.error(format.error(err.message));
      if (serverManager) await serverManager.stop();
      process.exit(1);
    }

    if (serverManager) await serverManager.stop();
    process.exit(0);
  }

  // Start the interactive TUI app
  const { InteractiveApp } = await import('./tui/app.js');
  const app = new InteractiveApp({
    agent,
    modelsConfig,
    warnings: [staleBuildWarning()].filter((w): w is string => w !== undefined),
  });

  await app.start();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(format.error(err.message));
    process.exit(1);
  });
}
