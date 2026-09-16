# dsh-lite: Lean Terminal Coding Harness for Local llama.cpp Models & DeepSeek

An ultra-lightweight, standalone interactive terminal AI coding assistant built for local `llama.cpp` models (GGUFs) and DeepSeek, inspired by Claude Code and `pi-lite`.

Zero Electron, zero React, zero browser dependencies. Instant startup (<40ms).

---

## Key Features

- **Fully Local via `llama.cpp`**:
  - Automatically loads models and launch flags from `models.yml` (e.g. models in `~/Documents/GGUFs`).
  - Automatically manages `llama-server` process lifecycle (spawns with `--ctx-size`, `--n-gpu-layers 99`, `--flash-attn on`, checks health, reuses active servers, and cleans up on exit).
- **KV Cache-Preserving Prompting**:
  - Byte-identical system prompt across the entire session to ensure `llama-server` reuses KV cache on every turn without prompt re-evaluation delays.
- **Smart Context Window Trimming**:
  - Trims to 60% of budget in one batch when context exceeds `contextWindow - maxTokens`.
  - Elides verbose tool output, then drops whole old turns, then truncates what is still oversized,
    so a request never exceeds the window the server was launched with. Tool schemas count against
    the same budget. A turn that cannot fit is reported rather than sent.
- **Dual Sampling Modes**:
  - `thinking`: For reasoning models, sending `enable_thinking: true` to templates that accept it.
  - `instruct`: Instruct sampling with presence penalty, and thinking switched off.
  - Switchable at any time with `/mode`; each model's `sampling:` block is re-read per mode.
  - `temperature`, `top_p`, `top_k`, `min_p` and `presence_penalty` all reach a llama.cpp server.
    The DeepSeek cloud API is sent only the parameters it accepts.
- **Essential Coding Tools**:
  - `bash`: Shell execution in its own process group. Runs without approval, the way an agent should; Esc kills the command and everything it started. Output keeps the end and spills the full log to a file the model can read.
  - `view_file`: File reader with line numbers and slice ranges. Streams, so a window deep inside a
    large log costs that window rather than the file; binary files are reported, not dumped.
  - `edit_file`: Surgical search-and-replace with **colorized unified diffs** (`+` green, `-` red).
  - `write_file`: File creation and atomic overwrites.
  - `list_dir`: Directory listing with file size badges.
  - `grep_search`: Fast regex/text search across the workspace, cancellable with Esc.
  - `web_search`: Web search. Uses DuckDuckGo by default; set `TAVILY_API_KEY` or `BRAVE_API_KEY`
    for a keyed backend that returns fuller page content and is not rate-limited.
  - `web_fetch`: Reads a page as clean Markdown. Works against local development servers
    (`http://localhost:3000`) as well as public sites.
- **Durable JSONL Session Persistence**:
  - Stores conversations under `.dsh/sessions/` with `--resume` and `/resume <id>` support.
  - Writes a `.gitignore` in `.dsh/` so transcripts never appear as untracked files in your repository.
- **Interactive Slash Commands**:
  - `/agent`, `/plan`, `/chat`, `/voice`: Switch interaction mode. Agent has every tool; plan has
    the read-only ones; chat has web search and fetch, but nothing that touches the workspace;
    voice can read, search, write new files and edit existing ones, but has no shell.
  - `/voice`: Switch to voice mode and open the voice socket. See **Voice control** below.
  - `/cd <path>`: Move the workspace tools work in, without restarting the model or server. The
    conversation and its transcript follow you to the new directory.
  - `/model [name]`: Switch between local GGUFs (restarts or reuses `llama-server`) or cloud DeepSeek.
  - `/mode [thinking|instruct]`: Toggle sampling mode, including the model's thinking switch.
  - `/serve`: Host the selected model for other machines, with live server logs.
  - `/disconnect` (`/stop`): Stop `llama-server` and unload the model without exiting.
  - `/clear` (`/new`): Clear the screen and start a new session.
  - `/resume [id]`: Resume a saved session, or the most recent one when no id is given.
  - `/quit` (`/exit`): Clean exit.

---

## Configuration (`models.yml`)

`dsh-lite` searches for `models.yml` in:
1. `--models <path>` or `$DSH_MODELS`
2. `./models.yml` in the directory `dsh` was started from
3. `~/.dsh/models.yml`
4. `models.yml` beside this install, so `dsh` finds your models from any directory
5. `models.example.yml` beside this install, the checked-in template

Your own `models.yml` is git-ignored, because model paths differ per machine. Start from the
template:

```sh
cp models.example.yml models.yml
```

A `contextWindow` larger than the `--ctx-size` in the same entry's `launchArgs` is lowered to match at
startup, with a notice, because the server cannot honour more than it was launched with.

Example `models.yml`:

```yaml
providers:
  llamacpp:
    baseUrl: http://localhost:8080/v1
    modelDir: ~/GGUFs
    llamaServer: llama-server
    models:
      - id: Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf
        name: Qwen3-4B-Instruct
        reasoning: false
        contextWindow: 65536
        maxTokens: 16384
        launchArgs:
          - "--port"
          - "8080"
          - "--ctx-size"
          - "65536"
          - "--n-gpu-layers"
          - "99"
          - "--flash-attn"
          - "on"

      - id: Qwen3.5-4B/Qwen3.5-4B-Q6_K.gguf
        name: Qwen3.5-4B-Q6_K
        reasoning: true
        contextWindow: 131072
        maxTokens: 16384
        launchArgs:
          - "--port"
          - "8080"
          - "--ctx-size"
          - "131072"
          - "--n-gpu-layers"
          - "99"
          - "--flash-attn"
          - "on"
```

---

## Voice control

`/voice` opens a unix domain socket at `~/.dsh/input.sock` (owner-only, `0600`) and switches to a
mode with no shell, where `write_file` refuses to replace an existing file. A shell cannot be made
non-destructive by filtering commands, so voice mode simply has none; `edit_file` is the only way to
change a file that already exists.

The socket speaks newline-delimited JSON, so a separate daemon owns the microphone, wake word,
endpointing and transcription, and the harness only ever sees text.

Sent to dsh:

| Line | Effect |
| --- | --- |
| `{"type":"text","text":"..."}` | Appends to the input line. Does **not** submit, so a misheard word can be seen first. |
| `{"type":"submit"}` | Submits the input line, as if Enter were pressed. |
| `{"type":"abort"}` | Aborts a running turn, or clears the pending line. The Escape key. |

Sent back, for the daemon to speak:

| Line | When |
| --- | --- |
| `{"type":"ack"}` | A turn started. |
| `{"type":"done","summary":"..."}` | A turn finished, condensed to one or two sentences. |
| `{"type":"error","message":"..."}` | A line could not be understood. The connection stays up. |

A daemon implementing the speaking end lives in [`voice/`](voice/README.md): openWakeWord for the
trigger, Silero VAD for endpointing, and MLX Whisper loaded per clip and dropped after.

Drive it without a daemon:

```sh
printf '{"type":"text","text":"write a haiku about llamas"}\n{"type":"submit"}\n' \
  | nc -U ~/.dsh/input.sock
```

Leaving voice mode closes the socket and removes the file, so only the window you asked for voice in
is driven by it. A second dsh entering voice mode while another holds the socket is refused.

## Environment variables

| Variable | Effect |
| --- | --- |
| `DSH_MODELS` | Path to a `models.yml`, ahead of every other location. |
| `DSH_MAX_STEPS` | Tool steps allowed per turn; `0` is unlimited. Default 100. |
| `DSH_STREAM_IDLE_TIMEOUT_MS` | Silence allowed mid-request before the model is treated as stalled. Default 300000. |
| `DSH_WEB_TIMEOUT_MS` | Deadline for `web_fetch` and `web_search` requests. Defaults 30000 and 20000. |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` | Use a keyed search backend instead of scraping DuckDuckGo. |
| `DEEPSEEK_API_KEY` | Key for the cloud `deepseek-chat` and `deepseek-reasoner` models. |
| `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` | Defaults for the cloud endpoint and model. |
| `DSH_DEBUG` | Print stack traces for startup errors. |

## Quick Start

### 1. List Available Models

```bash
cd dsh-lite
./bin/dsh.js --list-models
```

### 2. Run Interactive REPL with a Local Model

```bash
# Start with default local model in models.yml
./bin/dsh.js

# Or choose a specific model from models.yml
./bin/dsh.js -m Qwen3-4B-Instruct

# Or start in instruct mode
./bin/dsh.js -m Qwen3-4B-Instruct --mode instruct
```

### 3. Run with Cloud DeepSeek API (Optional)

```bash
export DEEPSEEK_API_KEY="sk-your-key"

./bin/dsh.js -m deepseek-chat
# Or with DeepSeek-R1 cloud reasoning:
./bin/dsh.js -m deepseek-reasoner
```

### 4. One-Shot Task Mode

```bash
./bin/dsh.js -m Qwen3-4B "run tests and fix any failing assertions"
```

### 5. Resuming Sessions

```bash
./bin/dsh.js --resume
```

---

## Development & Testing

```bash
# Run unit tests (16 tests covering tools, diffs, models.yml, and context trimming)
npm test

# Build TypeScript
npm run build
```
