import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import type { ToolDefinition } from './types.js';
import { IGNORE_DIRS } from './ignore.js';

interface GrepArgs {
  query: string;
  path?: string;
  isRegex?: boolean;
  caseInsensitive?: boolean;
  maxMatches?: number;
}

export const grepSearchTool: ToolDefinition<GrepArgs, string> = {
  name: 'grep_search',
  description: 'Search for text or regular expressions across files in the workspace. Returns matching filenames and line numbers.',
  parameters: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'The search term or regex pattern.',
      },
      path: {
        type: 'string',
        description: 'Directory or file to search within (default: workspace root).',
      },
      isRegex: {
        type: 'boolean',
        description: 'Whether to treat query as a regular expression (default: false).',
      },
      caseInsensitive: {
        type: 'boolean',
        description: 'Case insensitive match (default: true).',
      },
      maxMatches: {
        type: 'number',
        description: 'Maximum matches to return (default: 50).',
      },
    },
    required: ['query'],
  },
  execute: async (
    { query, path: searchPath = '.', isRegex = false, caseInsensitive = true, maxMatches = 50 },
    context
  ) => {
    const fullPath = path.isAbsolute(searchPath) ? searchPath : path.resolve(context.cwd, searchPath);

    let regex: RegExp;
    try {
      const flags = caseInsensitive ? 'i' : '';
      if (isRegex) {
        regex = new RegExp(query, flags);
      } else {
        const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        regex = new RegExp(escaped, flags);
      }
    } catch (err: any) {
      return `Invalid regex query: ${err.message}`;
    }

    const matches: string[] = [];

    async function searchInFile(filePath: string, relPath: string) {
      if (matches.length >= maxMatches || context.abortSignal?.aborted) return;
      try {
        const stat = await fs.stat(filePath);
        if (stat.size > 2 * 1024 * 1024) return; // Skip files > 2MB

        const content = await fs.readFile(filePath, 'utf8');
        const lines = content.split(/\r?\n/);

        for (let i = 0; i < lines.length; i++) {
          if (matches.length >= maxMatches) break;
          const line = lines[i];
          if (regex.test(line)) {
            matches.push(`${relPath}:${i + 1}: ${line.trimEnd()}`);
          }
        }
      } catch {
        // Skip binary / unreadable files
      }
    }

    async function walk(dir: string, rel: string) {
      if (matches.length >= maxMatches || context.abortSignal?.aborted) return;
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        if (matches.length >= maxMatches || context.abortSignal?.aborted) break;
        if (IGNORE_DIRS.has(entry.name)) continue;

        const relEntry = rel ? path.join(rel, entry.name) : entry.name;
        const entryFull = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          await walk(entryFull, relEntry);
        } else if (entry.isFile()) {
          await searchInFile(entryFull, relEntry);
        }
      }
    }

    try {
      const stat = await fs.stat(fullPath);
      if (stat.isFile()) {
        await searchInFile(fullPath, searchPath);
      } else {
        await walk(fullPath, '');
      }

      if (context.abortSignal?.aborted) {
        const found = matches.length > 0 ? `\n${matches.join('\n')}` : '';
        return `Search cancelled by user after ${matches.length} matches.${found}`;
      }

      if (matches.length === 0) {
        return `No matches found for "${query}".`;
      }

      let output = matches.join('\n');
      if (matches.length >= maxMatches) {
        output += `\n... [truncated at ${maxMatches} matches]`;
      }
      return output;
    } catch (err: any) {
      return `Error searching in ${searchPath}: ${err.message}`;
    }
  },
};
