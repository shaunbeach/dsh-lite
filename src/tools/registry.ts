import type OpenAI from 'openai';
import type { ToolDefinition, ToolExecutionContext, ToolExecutionResult } from './types.js';
import { bashTool } from './bash.js';
import { viewFileTool, writeFileTool } from './fs.js';
import { editFileTool } from './edit.js';
import { listDirTool } from './glob.js';
import { grepSearchTool } from './grep.js';
import { webSearchTool } from './web_search.js';
import { webFetchTool } from './web_fetch.js';

export class ToolRegistry {
  private tools: Map<string, ToolDefinition> = new Map();

  constructor() {
    this.register(bashTool);
    this.register(viewFileTool);
    this.register(writeFileTool);
    this.register(editFileTool);
    this.register(listDirTool);
    this.register(grepSearchTool);
    this.register(webSearchTool);
    this.register(webFetchTool);
  }

  public register(tool: ToolDefinition) {
    this.tools.set(tool.name, tool);
  }

  public getOpenAITools(): OpenAI.Chat.ChatCompletionTool[] {
    return Array.from(this.tools.values()).map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    }));
  }

  public async execute(
    name: string,
    rawArgs: string,
    toolCallId: string,
    context: ToolExecutionContext
  ): Promise<ToolExecutionResult> {
    const tool = this.tools.get(name);
    if (!tool) {
      return {
        toolCallId,
        name,
        result: `Unknown tool: ${name}`,
        isError: true,
      };
    }

    let parsedArgs: any = {};
    try {
      if (rawArgs.trim()) {
        parsedArgs = JSON.parse(rawArgs);
      }
    } catch (err: any) {
      return {
        toolCallId,
        name,
        result: `Failed to parse arguments for ${name}: ${err.message}`,
        isError: true,
      };
    }

    try {
      const output = await tool.execute(parsedArgs, context);

      // Handle special edit result object with diff
      if (name === 'edit_file' && typeof output === 'object' && output !== null && 'diff' in output) {
        return {
          toolCallId,
          name,
          result: output.message,
          diff: output.diff,
          isError: output.isError,
        };
      }

      return {
        toolCallId,
        name,
        result: typeof output === 'string' ? output : JSON.stringify(output, null, 2),
        isError: false,
      };
    } catch (err: any) {
      return {
        toolCallId,
        name,
        result: `Tool execution error: ${err.message}`,
        isError: true,
      };
    }
  }
}
