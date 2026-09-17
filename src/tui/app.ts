import {
  CombinedAutocompleteProvider,
  Container,
  Editor,
  matchesKey,
  ProcessTerminal,
  type SelectItem,
  SelectList,
  TuiMainScreen,
} from '@earendil-works/pi-tui';
import { Agent, type AgentTurnCallbacks, type InteractionMode } from '../agent.js';
import { estimateMessageTokens } from '../context.js';
import { getLocalIpAddress, LlamaServerManager, serverPort, stopServerOnExit } from '../llm/server.js';
import type { LiteModel, ModelsConfig } from '../config/models.js';
import { editorTheme, selectListTheme, style } from './theme.js';
import {
  type AiStatus,
  AssistantView,
  BannerView,
  formatFooter,
  formatTokens,
  Line,
  NoticeView,
  ServeView,
  ToolView,
  UserView,
} from './components.js';
import { COMMANDS, type CommandName, parseCommand, slashCommands } from './commands.js';
import { summariseForSpeech, VoiceSocket } from '../voice/socket.js';
import * as fs from 'node:fs';
import * as path from 'node:path';

export interface InteractiveAppOptions {
  agent: Agent;
  modelsConfig?: ModelsConfig;
  /** Startup problems to show once, alongside any the model catalogue reported. */
  warnings?: string[];
}

export class InteractiveApp {
  private agent: Agent;
  private modelsConfig?: ModelsConfig;
  private serverManager: LlamaServerManager;

  private tui: TuiMainScreen;
  private bannerView: BannerView;
  private chat: Container = new Container();
  private status: Line = new Line();
  private editorSlot: Container = new Container();
  private footer: Line = new Line();
  private editor: Editor;

  private isBusy = false;
  private isServing = false;
  private isSwitchingModel = false;
  private modelSwitchAbort?: AbortController;
  private voiceSocket?: VoiceSocket;
  private aiStatus: AiStatus = 'idle';
  private turnTimer?: NodeJS.Timeout;
  private turnStarted?: number;
  private serveAbortController?: AbortController;
  private abortController?: AbortController;
  private finishPromiseResolve = () => {};

  constructor(options: InteractiveAppOptions) {
    this.agent = options.agent;
    this.modelsConfig = options.modelsConfig;

    this.serverManager = new LlamaServerManager();
    stopServerOnExit(this.serverManager);

    this.tui = new TuiMainScreen(new ProcessTerminal());
    this.editor = new Editor(this.tui, editorTheme, { paddingX: 1 });

    const models = this.modelsConfig?.models ?? [];
    this.editor.setAutocompleteProvider(
      new CombinedAutocompleteProvider(slashCommands(models) as any, this.agent.cwd)
    );

    this.editor.onSubmit = (text) => this.submit(text);
    this.editorSlot.addChild(this.editor);

    const initialModelName = this.agent.isModelLoaded()
      ? (this.agent.activeModel?.name || this.agent.client.model)
      : undefined;
    const initialMode = initialModelName ? this.agent.client.mode : undefined;

    this.bannerView = new BannerView({
      version: '0.2.0',
      cwd: this.agent.cwd,
      modelName: initialModelName,
      mode: initialMode,
    });
    this.chat.addChild(this.bannerView);

    for (const warning of [...(options.warnings ?? []), ...(this.modelsConfig?.warnings ?? [])]) {
      this.chat.addChild(new NoticeView(warning));
    }

    this.tui.addChild(this.chat);
    this.tui.addChild(this.status);
    this.tui.addChild(this.editorSlot);
    this.tui.addChild(this.footer);
    this.tui.setFocus(this.editor);

    this.tui.addInputListener((data) => {
      const isEscape = matchesKey(data, 'escape') || matchesKey(data, 'esc') || data === '\x1b';
      const isCtrlC = matchesKey(data, 'ctrl+c');

      if (this.isServing) {
        if (isEscape || isCtrlC) {
          void this.stopServe();
          return { consume: true };
        }
      }

      if (this.isSwitchingModel) {
        if (isEscape || isCtrlC) {
          this.modelSwitchAbort?.abort();
          this.setStatus('Cancelling model load...');
          return { consume: true };
        }
      }

      if (isEscape || isCtrlC) {
        if (this.isBusy) {
          this.abortController?.abort();
          this.setStatus('Aborting operation...');
          return { consume: true };
        }
        if (isCtrlC) {
          if (this.editor.getText().length > 0) {
            this.editor.setText('');
            this.tui.requestRender();
          } else {
            this.tui.stop();
            this.finishPromiseResolve();
          }
          return { consume: true };
        }
        if (isEscape) {
          if (this.editor.getText().length > 0) {
            this.editor.setText('');
            this.tui.requestRender();
            return { consume: true };
          }
        }
      }
      return undefined;
    });

    // Initial footer setup (DO NOT start llama-server here!)
    this.updateFooter();
  }

  public async start(): Promise<void> {
    const donePromise = new Promise<void>((resolve) => {
      this.finishPromiseResolve = resolve;
    });

    this.tui.terminal.setTitle('dsh');
    this.tui.start();
    this.updateFooter();

    await donePromise;
    this.stopTurnTimer();
    await this.closeVoiceSocket();
    await this.serverManager.stop();
  }

  private startTurnTimer() {
    this.stopTurnTimer();
    this.turnTimer = setInterval(() => {
      if (this.turnStarted !== undefined) {
        const elapsed = Date.now() - this.turnStarted;
        this.updateFooter(elapsed);
      }
    }, 250);
  }

  private stopTurnTimer() {
    if (this.turnTimer) {
      clearInterval(this.turnTimer);
      this.turnTimer = undefined;
    }
  }

  private updateFooter(turnDurationMs?: number) {
    const isLoaded = this.agent.isModelLoaded();
    const modelName = isLoaded
      ? (this.agent.activeModel?.name || this.agent.client.model)
      : undefined;
    const mode = isLoaded ? this.agent.client.mode : undefined;
    const interactionMode = this.agent.interactionMode;
    const usedTokens =
      this.agent.lastTurnMetrics?.totalTokens ??
      (isLoaded ? this.agent.messages.reduce((acc, m) => acc + estimateMessageTokens(m), 0) : 0);
    const contextWindow = isLoaded ? this.agent.contextManager.contextWindow : undefined;
    const tokensPerSecond = this.agent.lastTurnMetrics?.predictedPerSecond;

    this.footer.setText(
      formatFooter({
        modelName,
        mode,
        interactionMode,
        aiStatus: this.isServing ? 'serving' : this.aiStatus,
        cwd: this.agent.cwd,
        usedTokens,
        contextWindow,
        tokensPerSecond,
        turnDurationMs,
        isRunning: this.isBusy,
      })
    );
    this.tui.requestRender();
  }

  private setStatus(msg?: string) {
    this.status.setText(msg ? style.yellow(` ${msg}`) : '');
    this.tui.requestRender();
  }

  private async submit(text: string) {
    const input = text.trim();
    if (!input) return;

    this.editor.setText('');
    this.editor.addToHistory(input);

    const command = parseCommand(input);
    if (command) {
      this.runCommand(command.name, command.args);
      return;
    }

    if (this.isBusy) return;
    if (this.isSwitchingModel) {
      this.setStatus('Still loading the model — press Esc to cancel.');
      return;
    }
    await this.runPrompt(input);
  }

  private async runPrompt(input: string) {
    if (!this.agent.isModelLoaded()) {
      this.setStatus('No model loaded. Please select a model with /model');
      this.pickModel();
      return;
    }

    this.isBusy = true;
    this.editor.disableSubmit = true;
    this.abortController = new AbortController();

    // If active model is a local model, ensure llama-server is ready before running prompt
    if (this.agent.activeModel) {
      this.setStatus(`Checking local server for ${this.agent.activeModel.name}...`);
      try {
        await this.serverManager.ensure(
          this.agent.activeModel,
          (status) => {
            this.setStatus(status);
          },
          this.abortController.signal
        );
        this.setStatus(undefined);
      } catch (err: any) {
        this.setStatus(undefined);
        this.chat.addChild(new UserView(input));
        if (this.abortController.signal.aborted) {
          this.chat.addChild(new NoticeView('operation aborted by user'));
        } else {
          const errView = new AssistantView();
          errView.finish(`Failed to start llama-server: ${err.message}`);
          this.chat.addChild(errView);
        }
        this.isBusy = false;
        this.editor.disableSubmit = false;
        this.editorSlot.clear();
        this.editorSlot.addChild(this.editor);
        this.tui.setFocus(this.editor);
        this.tui.requestRender();
        return;
      }
    }

    this.chat.addChild(new UserView(input));
    this.tui.requestRender();

    this.voiceSocket?.broadcast({ type: 'ack' });

    const turnStarted = Date.now();
    this.turnStarted = turnStarted;
    this.aiStatus = this.agent.client.mode === 'thinking' ? 'thinking' : 'working';
    this.startTurnTimer();
    this.updateFooter(0);

    let currentAssistantView: AssistantView | undefined;
    const activeToolViews = new Map<string, ToolView>();

    const callbacks: AgentTurnCallbacks = {
      onAssistantStart: () => {
        currentAssistantView?.finish();
        currentAssistantView = new AssistantView();
        this.chat.addChild(currentAssistantView);
        this.aiStatus = this.agent.client.mode === 'thinking' ? 'thinking' : 'working';
        this.updateFooter(Date.now() - turnStarted);
        this.tui.requestRender();
      },
      onReasoningDelta: (delta: string) => {
        if (!currentAssistantView) {
          currentAssistantView = new AssistantView();
          this.chat.addChild(currentAssistantView);
        }
        currentAssistantView.appendThinking(delta);
        if (this.aiStatus !== 'thinking') {
          this.aiStatus = 'thinking';
          this.updateFooter(Date.now() - turnStarted);
        }
        this.tui.requestRender();
      },
      onContentDelta: (delta: string) => {
        if (!currentAssistantView) {
          currentAssistantView = new AssistantView();
          this.chat.addChild(currentAssistantView);
        }
        currentAssistantView.appendContent(delta);
        if (this.aiStatus !== 'working') {
          this.aiStatus = 'working';
          this.updateFooter(Date.now() - turnStarted);
        }
        this.tui.requestRender();
      },
      onAssistantEnd: () => {
        currentAssistantView?.finish();
        this.tui.requestRender();
      },
      onStepComplete: () => {
        this.updateFooter(Date.now() - turnStarted);
      },
      onToolStart: (id: string, name: string, argsSummary: string) => {
        currentAssistantView?.finish();
        const toolView = new ToolView(name, argsSummary);
        activeToolViews.set(id, toolView);
        this.chat.addChild(toolView);
        if (this.aiStatus !== 'working') {
          this.aiStatus = 'working';
          this.updateFooter(Date.now() - turnStarted);
        }
        this.tui.requestRender();
      },
      onToolEnd: (id: string, name: string, execution: { result: string; isError: boolean; diff?: string }) => {
        const toolView = activeToolViews.get(id);
        if (toolView) {
          toolView.setResult(execution.result, execution.isError, execution.diff);
        }
        this.updateFooter(Date.now() - turnStarted);
        this.tui.requestRender();
      },
      onContextTrimmed: (summary: string) => {
        this.chat.addChild(new NoticeView(`context trimmed: ${summary}`));
        this.tui.requestRender();
      },
      onNotice: (message: string) => {
        this.chat.addChild(new NoticeView(message));
        this.tui.requestRender();
      },
      onCancelled: () => {
        currentAssistantView?.finish();
        this.chat.addChild(new NoticeView('operation aborted by user'));
        this.setStatus(undefined);
        this.tui.requestRender();
      },
    };

    try {
      await this.agent.runTurn(input, callbacks, this.abortController.signal);
    } catch (err: any) {
      if (this.abortController?.signal.aborted || err.name === 'AbortError' || err.name === 'APIUserAbortError') {
        // Aborted by user - already presented via onCancelled
      } else {
        if (!currentAssistantView) {
          currentAssistantView = new AssistantView();
          this.chat.addChild(currentAssistantView);
        }
        currentAssistantView.finish(err.message);
      }
    } finally {
      this.stopTurnTimer();
      currentAssistantView?.finish();
      this.isBusy = false;
      this.aiStatus = 'idle';
      this.turnStarted = undefined;
      this.abortController = undefined;
      this.editor.disableSubmit = false;
      this.editorSlot.clear();
      this.editorSlot.addChild(this.editor);
      this.editor.invalidate();
      this.tui.setFocus(this.editor);
      const turnDurationMs = Date.now() - turnStarted;
      this.updateFooter(turnDurationMs);
      this.setStatus(undefined);
      this.tui.requestRender();

      if (this.voiceSocket?.isOpen) {
        const reply = [...this.agent.messages].reverse().find(m => m.role === 'assistant' && m.content)?.content;
        this.voiceSocket.broadcast({ type: 'done', summary: summariseForSpeech(reply ?? '') });
      }
    }
  }

  private runCommand(name: CommandName, args: string) {
    switch (name) {
      case 'cd':
        void this.changeDirectory(args);
        break;

      case 'project':
        void this.createProject(args);
        break;

      case 'model':
        if (args) void this.switchModel(args);
        else this.pickModel();
        break;

      case 'mode':
        if (args === 'thinking' || args === 'instruct') {
          this.agent.client.setMode(args);
        } else {
          const next = this.agent.client.mode === 'thinking' ? 'instruct' : 'thinking';
          this.agent.client.setMode(next);
        }
        if (this.agent.isModelLoaded()) {
          const modelName = this.agent.activeModel?.name || this.agent.client.model;
          this.bannerView.setModel(modelName, this.agent.client.mode);
        }
        this.updateFooter();
        break;

      case 'agent':
      case 'plan':
      case 'chat':
      case 'voice':
        void this.switchInteractionMode(name);
        break;

      case 'serve':
        this.serveModel();
        break;

      case 'disconnect':
        void this.disconnectServer();
        break;

      case 'clear':
        this.chat.clear();
        this.chat.addChild(this.bannerView);
        this.agent.clearHistory();
        this.updateFooter();
        this.announce('Workspace cleared.');
        break;

      case 'resume':
        void this.resumeSession(args);
        break;

      case 'quit':
        this.tui.stop();
        this.finishPromiseResolve();
        break;
    }
  }

  /**
   * Tells a listening voice daemon how a command turned out.
   *
   * Only turns report themselves, through the agent loop. A command produces no assistant reply to
   * summarise, so without this a spoken "clear the workspace" would be carried out in silence.
   */
  private announce(summary: string): void {
    this.voiceSocket?.broadcast({ type: 'done', summary });
  }

  /**
   * Creates a directory, moves into it, and starts a fresh conversation.
   *
   * One command rather than three, because it is one intent: the point of starting a project is to
   * be working in it with nothing carried over. Doing it in pieces leaves half-states when a step
   * fails, and "start a new project" is a single thing to say.
   */
  private async createProject(name: string) {
    if (this.isBusy || this.isServing || this.isSwitchingModel) {
      this.setStatus('Busy — wait for the current operation to finish.');
      return;
    }

    const folder = name.trim();
    if (!folder) {
      this.setStatus('Usage: /project <name>');
      return;
    }
    // One directory inside the workspace, not a path: a spoken name should never be able to reach
    // up and out of where the work is happening.
    if (folder.includes('/') || folder.includes('\\') || folder.startsWith('.')) {
      const problem = `A project name cannot contain a path: ${folder}`;
      this.setStatus(problem);
      this.announce(problem);
      return;
    }

    const target = path.resolve(this.agent.cwd, folder);
    try {
      if (fs.existsSync(target)) {
        const problem = `${folder} already exists`;
        this.setStatus(problem);
        this.announce(`${problem}. Nothing was created.`);
        return;
      }
      fs.mkdirSync(target, { recursive: true });

      // Cleared before the move, so the empty conversation is what follows into the new workspace.
      this.agent.clearHistory();
      await this.agent.setCwd(target);

      this.bannerView.setCwd(target);
      this.editor.setAutocompleteProvider(
        new CombinedAutocompleteProvider(slashCommands(this.modelsConfig?.models ?? []) as any, target)
      );
      this.chat.clear();
      this.chat.addChild(this.bannerView);
      this.chat.addChild(new NoticeView(`New project in ${target}`));
      this.setStatus(undefined);
      this.updateFooter();
      this.tui.requestRender();
      this.announce(`Project ready. We are in ${folder}.`);
    } catch (err: any) {
      const problem = `Could not create ${folder}: ${err.message}`;
      this.setStatus(problem);
      this.announce(problem);
    }
  }

  /**
   * Points the workspace at another directory. The model and llama-server stay as they are; only
   * what the tools resolve paths against changes, along with everything on screen that names it.
   */
  /**
   * Switches interaction mode, opening the voice socket on the way into voice mode and closing it on
   * the way out. Tying the socket to the mode settles which window a daemon drives: the one where
   * voice was asked for, rather than whichever happened to start last.
   */
  private async switchInteractionMode(mode: InteractionMode) {
    const leavingVoice = this.agent.interactionMode === 'voice' && mode !== 'voice';
    if (leavingVoice) await this.closeVoiceSocket();

    this.agent.setInteractionMode(mode);

    if (mode === 'voice' && !this.voiceSocket?.isOpen) {
      const socket = new VoiceSocket();
      try {
        await socket.open({
          onText: (text) => this.onVoiceText(text),
          onSubmit: () => this.onVoiceSubmit(),
          onAbort: () => this.onVoiceAbort(),
        });
        this.voiceSocket = socket;
        this.chat.addChild(new NoticeView(`Voice mode. Listening on ${socket.address}`));
      } catch (err: any) {
        this.chat.addChild(new NoticeView(`Voice mode without a socket: ${err.message}`));
      }
    } else {
      this.chat.addChild(new NoticeView(`Switched to ${mode} mode`));
    }

    this.updateFooter();
    this.tui.requestRender();
    if (mode !== 'voice') this.announce(`${mode} mode.`);
  }

  private async closeVoiceSocket(): Promise<void> {
    const socket = this.voiceSocket;
    this.voiceSocket = undefined;
    if (socket) await socket.close();
  }

  /** Dictated text lands in the editor rather than running, so a misheard word can be seen first. */
  private onVoiceText(text: string): void {
    const spoken = text.trim();
    if (!spoken) return;
    const existing = this.editor.getText();
    this.editor.setText(existing ? `${existing} ${spoken}` : spoken);
    this.tui.requestRender();
  }

  private onVoiceSubmit(): void {
    const pending = this.editor.getText().trim();
    if (!pending) return;
    void this.submit(pending);
  }

  /** Barge-in. Mirrors Escape: stop a running turn, or clear what is waiting to be sent. */
  private onVoiceAbort(): void {
    if (this.isBusy) {
      this.abortController?.abort();
      this.setStatus('Aborting operation...');
      return;
    }
    if (this.isSwitchingModel) {
      this.modelSwitchAbort?.abort();
      this.setStatus('Cancelling model load...');
      return;
    }
    this.editor.setText('');
    this.tui.requestRender();
  }

  private async changeDirectory(target: string) {
    if (this.isBusy || this.isServing || this.isSwitchingModel) {
      this.setStatus('Busy — wait for the current operation to finish.');
      return;
    }
    if (!target) {
      this.setStatus(`Usage: /cd <path>   (currently ${this.agent.cwd})`);
      return;
    }

    try {
      const moved = await this.agent.setCwd(target);
      this.bannerView.setCwd(moved);
      // File completion is rooted at the workspace, so it has to be rebuilt for the new one.
      this.editor.setAutocompleteProvider(
        new CombinedAutocompleteProvider(slashCommands(this.modelsConfig?.models ?? []) as any, moved)
      );
      this.chat.addChild(new NoticeView(`Workspace is now ${moved}`));
      this.setStatus(undefined);
      this.updateFooter();
      this.tui.requestRender();
      this.announce(`Now in ${path.basename(moved)}.`);
    } catch (err: any) {
      this.setStatus(`Could not change directory: ${err.message}`);
      this.announce(`Could not change directory: ${err.message}`);
    }
  }

  private async resumeSession(sessionId: string) {
    if (this.isBusy || this.isServing || this.isSwitchingModel) {
      this.setStatus('Busy — wait for the current operation to finish.');
      return;
    }

    try {
      let target = sessionId;
      if (!target) {
        const sessions = await this.agent.sessionStore.listSessions();
        if (sessions.length === 0) {
          this.setStatus('No saved sessions to resume.');
          return;
        }
        target = sessions[0].id;
      }

      const restored = await this.agent.resume(target);
      this.agent.lastTurnMetrics = undefined;
      this.chat.clear();
      this.chat.addChild(this.bannerView);
      this.chat.addChild(
        new NoticeView(`Resumed ${target}: ${restored} messages restored (not replayed on screen).`)
      );
      this.setStatus(undefined);
      this.updateFooter();
      this.tui.requestRender();
    } catch (err: any) {
      this.setStatus(`Could not resume: ${err.message}`);
    }
  }

  private pickModel() {
    const models = this.modelsConfig?.models ?? [];
    const isLoaded = this.agent.isModelLoaded();
    const current = isLoaded ? (this.agent.activeModel?.name || this.agent.client.model) : undefined;

    const items: SelectItem[] = models.map((m) => ({
      value: m.name,
      label: current && m.name === current ? `${m.name} (active)` : m.name,
      description: `${m.reasoning ? 'thinking' : 'instruct'} · ctx ${formatTokens(m.contextWindow)}`,
    }));

    items.push({
      value: 'deepseek-chat',
      label: current === 'deepseek-chat' ? 'deepseek-chat [Cloud] (active)' : 'deepseek-chat [Cloud]',
      description: 'DeepSeek-V3 Official API',
    });

    items.push({
      value: 'deepseek-reasoner',
      label: current === 'deepseek-reasoner' ? 'deepseek-reasoner [Cloud] (active)' : 'deepseek-reasoner [Cloud]',
      description: 'DeepSeek-R1 Official API',
    });

    this.openPicker('Switch Model', items, async (name) => {
      await this.switchModel(name);
    });
  }

  private async switchModel(name: string) {
    if (this.isBusy || this.isServing || this.isSwitchingModel) {
      this.setStatus('Busy — wait for the current operation to finish.');
      return;
    }

    const models = this.modelsConfig?.models ?? [];
    const matched = models.find(
      (m) => m.name.toLowerCase() === name.toLowerCase() || m.name.toLowerCase().includes(name.toLowerCase())
    );

    if (!matched && !(name.startsWith('deepseek-') || name === 'cloud')) {
      this.setStatus(`No model matching "${name}"`);
      this.announce(`I could not find a model called ${name}.`);
      return;
    }

    if (matched) {
      this.agent.setModel(matched);
      this.agent.lastTurnMetrics = undefined;
      this.bannerView.setModel(matched.name, matched.mode || (matched.reasoning ? 'thinking' : 'instruct'));
      this.isSwitchingModel = true;
      this.modelSwitchAbort = new AbortController();
      this.editor.disableSubmit = true;
      this.updateFooter();
      this.setStatus(`Starting server for ${matched.name}... Press Esc to cancel.`);
      try {
        await this.serverManager.ensure(matched, (msg) => this.setStatus(msg), this.modelSwitchAbort.signal);
        this.setStatus(undefined);
        this.announce(`${matched.name} is loaded.`);
      } catch (err: any) {
        if (this.modelSwitchAbort.signal.aborted) {
          this.chat.addChild(new NoticeView(`Model load cancelled; ${matched.name} is not running.`));
          this.setStatus(undefined);
          this.announce('Model load cancelled.');
        } else {
          this.chat.addChild(new NoticeView(`Failed to start server for ${matched.name}: ${err.message}`));
          this.setStatus(undefined);
          this.announce(`Could not load ${matched.name}.`);
        }
      } finally {
        this.isSwitchingModel = false;
        this.modelSwitchAbort = undefined;
        this.editor.disableSubmit = false;
        this.tui.setFocus(this.editor);
        this.updateFooter();
        this.tui.requestRender();
      }
    } else if (name.startsWith('deepseek-') || name === 'cloud') {
      const modelName = name === 'cloud' ? 'deepseek-chat' : name;
      this.agent.activeModel = undefined;
      this.agent.lastTurnMetrics = undefined;
      this.agent.client.setEndpoint('https://api.deepseek.com', process.env.DEEPSEEK_API_KEY || '');
      // The DeepSeek cloud API rejects llama.cpp's sampling extensions.
      this.agent.client.configureModel({ model: modelName, extendedSampling: false });
      this.bannerView.setModel(modelName, this.agent.client.mode);
      this.updateFooter();
      this.announce(`${modelName} is selected.`);
    }
  }

  private serveModel() {
    if (this.isBusy || this.isServing || this.isSwitchingModel) return;
    const models = this.modelsConfig?.models ?? [];
    if (models.length === 0) {
      this.setStatus('No local models found in models.yml to serve.');
      return;
    }

    const items: SelectItem[] = models.map((m) => ({
      value: m.name,
      label: m.name,
      description: `${m.reasoning ? 'thinking' : 'instruct'} · ctx ${formatTokens(m.contextWindow)} · port ${serverPort(m.baseUrl)}`,
    }));

    this.openPicker('Select Model to Serve', items, (name) => {
      void this.startServe(name);
    });
  }

  private async startServe(modelName: string) {
    const model = this.modelsConfig?.models.find((m) => m.name === modelName);
    if (!model) {
      this.setStatus(`Model ${modelName} not found in models.yml`);
      return;
    }

    this.isServing = true;
    this.aiStatus = 'serving';
    this.serveAbortController = new AbortController();
    const port = serverPort(model.baseUrl);
    const localIp = getLocalIpAddress();

    const serveView = new ServeView({
      modelName: model.name,
      port,
      localUrl: `http://localhost:${port}/v1`,
      remoteUrl: `http://${localIp}:${port}/v1`,
    });
    this.chat.addChild(serveView);
    this.setStatus(`Starting host server for ${model.name}... Press Esc or Ctrl+C to stop.`);
    this.updateFooter();
    this.tui.requestRender();

    try {
      await this.serverManager.startHost(
        model,
        (line) => {
          serveView.addLogLine(line);
          this.tui.requestRender();
        },
        this.serveAbortController.signal
      );
      this.setStatus(`Serving ${model.name} on http://${localIp}:${port}/v1 · Press Esc or Ctrl+C to stop`);
    } catch (err: any) {
      if (this.serveAbortController?.signal.aborted) {
        // Aborted cleanly by user
      } else {
        serveView.addLogLine(`Error starting server: ${err.message}`);
        this.setStatus(`Failed to serve ${model.name}: ${err.message}`);
      }
      this.isServing = false;
      this.aiStatus = 'idle';
      this.serveAbortController = undefined;
      this.updateFooter();
      this.tui.requestRender();
    }
  }

  private async stopServe() {
    if (!this.isServing) return;
    this.setStatus('Stopping host server...');
    this.serveAbortController?.abort();
    await this.serverManager.stop();
    this.isServing = false;
    this.aiStatus = 'idle';
    this.serveAbortController = undefined;
    this.setStatus(undefined);
    this.chat.addChild(new NoticeView('Host server stopped.'));
    this.editorSlot.clear();
    this.editorSlot.addChild(this.editor);
    this.tui.setFocus(this.editor);
    this.updateFooter();
    this.tui.requestRender();
  }

  private async disconnectServer() {
    if (this.isSwitchingModel) {
      this.setStatus('Still loading a model — press Esc to cancel that first.');
      return;
    }

    if (this.isServing) {
      await this.stopServe();
      return;
    }

    if (!this.serverManager.isRunning && !this.agent.isModelLoaded()) {
      this.setStatus('No active model or server to disconnect.');
      return;
    }

    this.setStatus('Disconnecting and stopping llama-server...');
    await this.serverManager.stop();
    this.agent.activeModel = undefined;
    this.agent.client.configureModel({ model: '' });
    this.agent.lastTurnMetrics = undefined;
    this.bannerView.setModel(undefined, undefined);
    this.chat.addChild(new NoticeView('Disconnected: llama-server stopped.'));
    this.setStatus(undefined);
    this.updateFooter();
    this.tui.requestRender();
    this.announce('Server stopped.');
  }

  private openPicker(title: string, items: SelectItem[], onSelect: (value: string) => void) {
    const list = new SelectList(items, Math.min(items.length, 10), selectListTheme, {
      maxPrimaryColumnWidth: 64,
    });

    const close = () => {
      this.editorSlot.clear();
      this.editorSlot.addChild(this.editor);
      this.tui.setFocus(this.editor);
      this.tui.requestRender();
    };

    list.onSelect = (item) => {
      close();
      onSelect(item.value);
    };

    list.onCancel = () => close();

    this.editorSlot.clear();
    this.editorSlot.addChild(list);
    this.tui.setFocus(list);
    this.tui.requestRender();
  }
}
