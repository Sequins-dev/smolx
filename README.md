# smolx

Serve local HuggingFace LLMs on Apple Silicon over OpenAI- and Anthropic-compatible HTTP APIs, then point your favorite coding agent (Claude Code, Codex, aider, opencode, pi, crush) at the local endpoint.

Built on [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) for inference and [Hummingbird](https://github.com/hummingbird-project/hummingbird) for the server. macOS 14+ only.

## Install

Requires a full Xcode (or the standalone Metal Toolchain) for MLX shader compilation, plus [xcodegen](https://github.com/yonaskolb/XcodeGen):

```sh
brew install xcodegen
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
make install   # builds via xcodebuild, installs to ~/.local/bin
```

`make install` copies both the `smolx` binary and the sibling `mlx-swift_Cmlx.bundle` (containing `default.metallib`) — MLX needs the bundle next to the executable at runtime.

## Quickstart

```sh
# 1. Pull a model from HuggingFace (interactive picker on partial names)
smolx pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

# 2. Pick which model fills each tier
smolx config set smart qwen2.5-coder-7b-instruct-4bit
smolx config set fast  qwen2.5-coder-7b-instruct-4bit
smolx config set small qwen2.5-coder-7b-instruct-4bit

# 3. Run the server
smolx serve   # listens on 127.0.0.1:8080 by default

# 4. Launch an agent against it (in another terminal)
smolx run claude
smolx run codex
smolx run aider -- --message "summarise this README"
```

`smolx run <agent>` execs the agent CLI with the env vars / config files it needs to talk to the local server. Supported agents: `claude`, `codex`, `aider`, `opencode`, `pi`, `crush`.

## Commands

| Command | Purpose |
| --- | --- |
| `smolx serve` | Run the HTTP server. Flags: `--port`, `--bind`, `--auth-token`, `--memory-budget`, `--idle-timeout`, `--max-concurrent`. |
| `smolx pull <repo>` | Download an MLX-format HuggingFace repo into the local cache and register it. |
| `smolx models` | List installed models (`--json` for machine output). |
| `smolx rm <name>` | Remove a model from the registry (`--purge` to also delete cached files). |
| `smolx config {set,get,unset} <tier> [value]` | Manage the `smart` / `fast` / `small` tier mapping that `smolx run` resolves. |
| `smolx run <agent>` | Launch a coding agent CLI wired to the local server. |

See `smolx <command> --help` for details.

## HTTP API

The server exposes both OpenAI- and Anthropic-compatible endpoints out of the same process:

- `GET  /v1/models`
- `POST /v1/chat/completions` — OpenAI chat completions (streaming + non-streaming)
- `POST /v1/responses` — OpenAI Responses API
- `POST /v1/messages` — Anthropic Messages API (streaming + non-streaming)
- `GET  /healthz`

Bind to a non-loopback address (`--bind 0.0.0.0`) only with `--auth-token`; the server refuses to start otherwise.

## Development

```sh
make build   # swift build (no Metal — fine for non-inference code paths)
make test    # swift test
make run ARGS="serve --port 9000"
```

`swift build` compiles cleanly but produces a binary that can't run inference (no Metal shaders). Use `make xcode-build` or `make install` for anything that touches MLX at runtime. See the [Makefile](./Makefile) for the full set of targets.

## License

MIT.
