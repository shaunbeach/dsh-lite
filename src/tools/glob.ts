import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import type { ToolDefinition } from './types.js';
import { IGNORE_DIRS } from './ignore.js';

interface ListDirArgs {
  path?: string;
  recursive?: boolean;
  maxEntries?: number;
}

export const listDirTool: ToolDefinition<ListDirArgs, string> = {
  name: 'list_dir',
  description: 'List the contents of a directory. Shows directories and files with sizes.',
  parameters: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Directory path to list (defaults to workspace root).',
      },
      recursive: {
        type: 'boolean',
        description: 'Whether to list subdirectories recursively (default: false).',
      },
      maxEntries: {
        type: 'number',
        description: 'Maximum number of items to return (default: 100).',
      },
    },
  },
  execute: async ({ path: dirPath = '.', recursive = false, maxEntries = 100 }, context) => {
    const fullPath = path.isAbsolute(dirPath) ? dirPath : path.resolve(context.cwd, dirPath);

    try {
      const results: string[] = [];

      async function scan(current: string, relative: string) {
        if (results.length >= maxEntries || context.abortSignal?.aborted) return;

        const entries = await fs.readdir(current, { withFileTypes: true });
        // Sort directories first, then alphabetical
        entries.sort((a, b) => {
          if (a.isDirectory() && !b.isDirectory()) return -1;
          if (!a.isDirectory() && b.isDirectory()) return 1;
          return a.name.localeCompare(b.name);
        });

        for (const entry of entries) {
          if (results.length >= maxEntries || context.abortSignal?.aborted) break;
          if (IGNORE_DIRS.has(entry.name)) continue;

          const relPath = relative ? path.join(relative, entry.name) : entry.name;
          const entryFullPath = path.join(current, entry.name);

          if (entry.isDirectory()) {
            results.push(`📁 ${relPath}/`);
            if (recursive) {
              await scan(entryFullPath, relPath);
            }
          } else {
            try {
              const stat = await fs.stat(entryFullPath);
              const sizeKb = (stat.size / 1024).toFixed(1);
              results.push(`📄 ${relPath} (${sizeKb} KB)`);
            } catch {
              results.push(`📄 ${relPath}`);
            }
          }
        }
      }

      await scan(fullPath, '');

      if (context.abortSignal?.aborted) {
        return `Listing cancelled by user after ${results.length} entries.`;
      }

      if (results.length === 0) {
        return `Directory ${dirPath} is empty.`;
      }

      let output = results.join('\n');
      if (results.length >= maxEntries) {
        output += `\n... [truncated at ${maxEntries} entries]`;
      }
      return output;
    } catch (err: any) {
      return `Error listing directory ${dirPath}: ${err.message}`;
    }
  },
};
