import * as fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import * as path from 'node:path';
import type { ToolDefinition } from './types.js';

/** Bytes sampled to decide whether a file is text before any of it is shown to the model. */
const BINARY_SNIFF_BYTES = 8192;

/** A NUL byte in the first block is the usual sign of a binary file; utf8 text has none. */
async function looksBinary(filePath: string): Promise<boolean> {
  const handle = await fs.open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(BINARY_SNIFF_BYTES);
    const { bytesRead } = await handle.read(buffer, 0, BINARY_SNIFF_BYTES, 0);
    return buffer.subarray(0, bytesRead).includes(0);
  } finally {
    await handle.close();
  }
}

interface ViewFileArgs {
  path: string;
  offset?: number;
  limit?: number;
}

export const viewFileTool: ToolDefinition<ViewFileArgs, string> = {
  name: 'view_file',
  description: 'View the contents of a file with line numbers. You can specify offset (1-indexed start line) and limit (number of lines) to view specific sections.',
  parameters: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Relative or absolute file path to view.',
      },
      offset: {
        type: 'number',
        description: 'The 1-indexed line number to start reading from (default: 1).',
      },
      limit: {
        type: 'number',
        description: 'Maximum number of lines to read (default: 500).',
      },
    },
    required: ['path'],
  },
  execute: async ({ path: filePath, offset = 1, limit = 500 }, context) => {
    const fullPath = path.isAbsolute(filePath) ? filePath : path.resolve(context.cwd, filePath);
    try {
      const stats = await fs.stat(fullPath);
      if (stats.isDirectory()) {
        return `Error viewing file ${filePath}: it is a directory. Use list_dir instead.`;
      }
      if (await looksBinary(fullPath)) {
        return `${filePath} is a binary file (${stats.size} bytes). Inspect it with bash, for example \`file\` or \`xxd\`.`;
      }

      // Read line by line rather than loading the file: a multi-gigabyte log must cost the
      // requested window of lines, not its whole size in memory.
      const start = Math.max(1, offset);
      const end = start + Math.max(1, limit);
      const collected: string[] = [];
      let lineNumber = 0;
      let moreFollow = false;

      const input = createReadStream(fullPath, { encoding: 'utf8' });
      const reader = createInterface({ input, crlfDelay: Infinity });
      try {
        for await (const line of reader) {
          lineNumber++;
          if (lineNumber < start) continue;
          if (lineNumber >= end) {
            moreFollow = true;
            break;
          }
          collected.push(`${String(lineNumber).padStart(6, ' ')} | ${line}`);
          if (context.abortSignal?.aborted) break;
        }
      } finally {
        reader.close();
        input.destroy();
      }

      if (collected.length === 0) {
        return lineNumber === 0
          ? `File: ${filePath} is empty.`
          : `File: ${filePath} has ${lineNumber} lines; offset ${start} is past the end.`;
      }

      const last = start + collected.length - 1;
      const range = moreFollow
        ? `lines ${start}-${last}, more follow`
        : `lines ${start}-${last} of ${lineNumber}`;
      return `File: ${filePath} (${range})\n${collected.join('\n')}`;
    } catch (err: any) {
      return `Error viewing file ${filePath}: ${err.message}`;
    }
  },
};

interface WriteFileArgs {
  path: string;
  content: string;
}

export const writeFileTool: ToolDefinition<WriteFileArgs, string> = {
  name: 'write_file',
  description: 'Create a new file or completely overwrite an existing file with the specified content.',
  parameters: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Relative or absolute file path to write.',
      },
      content: {
        type: 'string',
        description: 'The complete contents to write into the file.',
      },
    },
    required: ['path', 'content'],
  },
  execute: async ({ path: filePath, content }, context) => {
    const fullPath = path.isAbsolute(filePath) ? filePath : path.resolve(context.cwd, filePath);
    try {
      await fs.mkdir(path.dirname(fullPath), { recursive: true });
      await fs.writeFile(fullPath, content, 'utf8');
      return `Successfully wrote ${Buffer.byteLength(content, 'utf8')} bytes to ${filePath}`;
    } catch (err: any) {
      return `Error writing file ${filePath}: ${err.message}`;
    }
  },
};
