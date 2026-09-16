import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';
import { load } from 'js-yaml';

export interface SamplingConfig {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  thinking?: boolean;
  reasoningEffort?: string;
  extra?: Record<string, any>;
}

export interface LiteModel {
  id: string;
  name: string;
  modelPath: string;
  baseUrl: string;
  llamaServer: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  launchArgs: string[];
  mode?: 'thinking' | 'instruct';
  sampling?: {
    thinking?: SamplingConfig;
    instruct?: SamplingConfig;
  };
}

export interface ModelsConfig {
  configPath: string;
  models: LiteModel[];
  /** Corrections applied to the file's values, for the caller to show the user. */
  warnings: string[];
}

function readNumericArg(launchArgs: string[], ...flags: string[]): number | undefined {
  for (const flag of flags) {
    const index = launchArgs.lastIndexOf(flag);
    if (index >= 0 && index + 1 < launchArgs.length) {
      const value = Number(launchArgs[index + 1]);
      if (Number.isFinite(value) && value > 0) return value;
    }
  }
  return undefined;
}

/**
 * Context llama-server will actually grant one request, or undefined when
 * launchArgs leave it to the server default. --ctx-size is the whole KV cache,
 * divided evenly across --parallel slots.
 */
export function effectiveServerContext(launchArgs: string[]): number | undefined {
  const ctxSize = readNumericArg(launchArgs, '--ctx-size', '--ctx_size', '-c');
  if (ctxSize === undefined) return undefined;
  const parallel = readNumericArg(launchArgs, '--parallel', '-np') ?? 1;
  return Math.floor(ctxSize / parallel);
}

export function resolveHomePath(p: string): string {
  if (p.startsWith('~/') || p === '~') {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}

/** A file beside this install, resolved from wherever dsh was started. */
function bundledPath(name: string): string {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', name);
}

/**
 * Locates the model catalogue, most specific source first: an explicit
 * --models path, DSH_MODELS, a models.yml in the directory dsh was started
 * from, the user's own ~/.dsh/models.yml, one beside this install, and finally
 * the checked-in example. Without the last two dsh finds no models at all when
 * run outside its own directory, which is the normal case.
 */
export function findModelsConfigPath(explicitPath?: string): string | undefined {
  if (explicitPath && fs.existsSync(explicitPath)) return explicitPath;
  if (process.env.DSH_MODELS && fs.existsSync(process.env.DSH_MODELS)) return process.env.DSH_MODELS;

  const candidates = [
    path.resolve(process.cwd(), 'models.yml'),
    path.join(os.homedir(), '.dsh', 'models.yml'),
    bundledPath('models.yml'),
    // A fresh clone has no models.yml of its own, since each machine's paths differ.
    bundledPath('models.example.yml'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }

  return undefined;
}

export function loadModelsConfig(explicitPath?: string): ModelsConfig | undefined {
  const configPath = findModelsConfigPath(explicitPath);
  if (!configPath) return undefined;

  try {
    const raw = fs.readFileSync(configPath, 'utf8');
    const parsed = load(raw) as any;
    if (!parsed || typeof parsed !== 'object') return undefined;

    const provider = parsed.providers?.llamacpp;
    if (!provider) return undefined;

    const baseUrl = provider.baseUrl || 'http://localhost:8080/v1';
    const rawModelDir = provider.modelDir || os.homedir();
    const modelDir = resolveHomePath(rawModelDir);
    const llamaServer = provider.llamaServer || 'llama-server';

    const warnings: string[] = [];
    const rawModels: any[] = Array.isArray(provider.models) ? provider.models : [];
    const models: LiteModel[] = rawModels.map((m: any) => {
      const id = m.id || m.name || 'unnamed-model';
      const name = m.name || path.basename(id, '.gguf');
      const modelPath = path.isAbsolute(id) ? id : path.join(modelDir, id);
      const launchArgs: string[] = Array.isArray(m.launchArgs) ? m.launchArgs.map(String) : [];

      // A contextWindow larger than the server's own window would let the
      // agent budget a prompt llama-server cannot accept, so the server wins.
      const declaredContext = m.contextWindow || 32768;
      const serverContext = effectiveServerContext(launchArgs);
      const contextWindow =
        serverContext !== undefined && declaredContext > serverContext ? serverContext : declaredContext;
      if (contextWindow !== declaredContext) {
        warnings.push(
          `${name}: contextWindow ${declaredContext} exceeds the ${serverContext} its launchArgs give llama-server; using ${contextWindow}.`
        );
      }

      return {
        id,
        name,
        modelPath,
        baseUrl: m.baseUrl || baseUrl,
        llamaServer: m.llamaServer || llamaServer,
        reasoning: Boolean(m.reasoning),
        contextWindow,
        maxTokens: m.maxTokens || 4096,
        launchArgs,
        mode: m.mode,
        sampling: m.sampling,
      };
    });

    return {
      configPath,
      models,
      warnings,
    };
  } catch (err: any) {
    console.error(`Error reading ${configPath}: ${err.message}`);
    return undefined;
  }
}
