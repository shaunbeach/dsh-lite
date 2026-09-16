import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { createTwoFilesPatch } from 'diff';
import type { ToolDefinition } from './types.js';

interface EditFileArgs {
  path: string;
  target: string;
  replacement: string;
}

export interface EditResult {
  message: string;
  diff?: string;
  isError: boolean;
}

export const editFileTool: ToolDefinition<EditFileArgs, EditResult> = {
  name: 'edit_file',
  description: 'Perform an exact text replacement in a file. The target text must exist uniquely in the file. Generates a unified diff of the change.',
  parameters: {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Relative or absolute file path to edit.',
      },
      target: {
        type: 'string',
        description: 'The exact string in the file to be replaced. Must be unique in the file.',
      },
      replacement: {
        type: 'string',
        description: 'The replacement text to insert in place of target.',
      },
    },
    required: ['path', 'target', 'replacement'],
  },
  execute: async ({ path: filePath, target, replacement }, context) => {
    const fullPath = path.isAbsolute(filePath) ? filePath : path.resolve(context.cwd, filePath);
    try {
      const originalContent = await fs.readFile(fullPath, 'utf8');

      const occurrences = originalContent.split(target).length - 1;
      if (occurrences === 0) {
        return {
          message: `Error: target text not found in ${filePath}. Verify exact whitespace and linebreaks.`,
          isError: true,
        };
      }
      if (occurrences > 1) {
        return {
          message: `Error: target text occurs ${occurrences} times in ${filePath}. Target must match exactly once. Include more surrounding lines to make it unique.`,
          isError: true,
        };
      }

      // The function form keeps the replacement literal. Passing it as a string would let `$&`,
      // `` $` ``, `$'` and `$1` act as substitution patterns and corrupt shell, regex or template code.
      const newContent = originalContent.replace(target, () => replacement);
      await fs.writeFile(fullPath, newContent, 'utf8');

      const diff = createTwoFilesPatch(
        `a/${filePath}`,
        `b/${filePath}`,
        originalContent,
        newContent,
        '',
        ''
      );

      return {
        message: `Successfully edited ${filePath}`,
        diff,
        isError: false,
      };
    } catch (err: any) {
      return {
        message: `Error editing file ${filePath}: ${err.message}`,
        isError: true,
      };
    }
  },
};
