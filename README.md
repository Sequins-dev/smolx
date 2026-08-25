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
smolx run opencode -- "summarise this README"
```

`smolx run <agent>` execs the agent CLI with the env vars / config files it needs to talk to the local smolx endpoint. Supported agents: `claude`, `codex`, `opencode`, `pi`, `crush`.

To use LM Studio, run a target-specific compatibility proxy in place of `serve`, then launch the agent normally:

```sh
# Terminal 1: local endpoint on 127.0.0.1:8080, forwarding to LM Studio
smolx proxy lm-studio --base-url http://192.168.1.55:1234

# Terminal 2: agents always connect to the local endpoint
smolx run codex
```

The proxy forwards OpenAI-compatible routes and translates target-specific wire-format differences. For LM Studio, Codex's `additional_tools` input item is promoted to the standard top-level `tools` array before forwarding. `smolx run` always reads the model catalog through the local endpoint, whether it is backed by `serve` or `proxy`.

### Run the proxy as a service

Generate a macOS user LaunchAgent after installing the release binary, then manage it with `launchctl`:

```sh
make install
smolx proxy launchd lm-studio --base-url http://192.168.1.55:1234

launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/dev.sequins.smolx.proxy.plist"
launchctl print gui/$(id -u)/dev.sequins.smolx.proxy
launchctl kickstart -k gui/$(id -u)/dev.sequins.smolx.proxy
launchctl bootout gui/$(id -u)/dev.sequins.smolx.proxy
```

The generator pins the `smolx` executable that runs it because launchd requires an absolute executable path. It writes `~/Library/LaunchAgents/dev.sequins.smolx.proxy.plist` with mode `0600` because the plist may contain proxy credentials. Logs go to `~/.smolx/logs/proxy.log` and `~/.smolx/logs/proxy.error.log`. To change the proxy configuration, boot out the service, regenerate the plist, and bootstrap it again. Pass `--output` to write the plist elsewhere.

## Commands

| Command | Purpose |
| --- | --- |
| `smolx serve` | Run the HTTP server. Flags: `--port`, `--bind`, `--auth-token`, `--keep-free`, `--idle-timeout`, `--max-concurrent`. |
| `smolx proxy <target>` | Run a local compatibility proxy to another model server. Currently supports `lm-studio`; pass its upstream URL with `--base-url`. |
| `smolx proxy launchd <target>` | Generate a native macOS LaunchAgent plist for the proxy. |
| `smolx pull <repo>` | Download an MLX-format HuggingFace repo into the local cache and register it. |
| `smolx models` | List installed models (`--json` for machine output). |
| `smolx rm <name>` | Remove a model from the registry (`--purge` to also delete cached files). |
| `smolx config {set,get,unset} <tier> [value]` | Manage the `smart` / `fast` / `small` tier mapping that `smolx run` resolves. |
| `smolx run <agent>` | Launch a coding agent CLI wired to the local `serve` or `proxy` endpoint. |

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
