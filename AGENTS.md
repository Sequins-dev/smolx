# Repository Guidelines

## Project Structure & Module Organization

`smolx` is a Swift Package Manager executable for macOS 14+.

- `Sources/smolx/Smolx.swift` contains the CLI entry point.
- `Sources/smolx/Commands/` implements ArgumentParser subcommands such as `serve`, `pull`, `models`, `config`, and `run`.
- `Sources/smolx/Server/` contains the Hummingbird HTTP server and route handlers.
- `Sources/smolx/API/` holds OpenAI, Anthropic, and shared domain translation types.
- `Sources/smolx/Providers/`, `Models/`, `Agents/`, `Config/`, and `Hub/` contain MLX integration, model registry logic, agent launch wiring, user paths/config, and HuggingFace download utilities.
- `Tests/smolxTests/` contains Swift Testing suites.

Keep new code near the feature it supports and avoid broad helpers unless they are shared by existing areas.

## Build, Test, and Development Commands

- `make build`: runs `swift build` for compile checks and non-inference paths.
- `make test`: runs `swift test` with Command Line Tools framework and rpath flags.
- `make run ARGS="serve --port 9000"`: runs the CLI through SwiftPM.
- `make xcode-build`: regenerates `smolx.xcodeproj`, then builds with Xcode so MLX Metal shaders are compiled.
- `make install`: builds via Xcode and installs `smolx` plus `mlx-swift_Cmlx.bundle` to `~/.local/bin`.

Use the Xcode build path for MLX inference. A plain SwiftPM binary should compile but not run inference.

## Coding Style & Naming Conventions

Use Swift 6 conventions with 4-space indentation and a 120-column line length, matching `.swift-format`. Types and suites use `UpperCamelCase`; methods, variables, enum cases, and test functions use `lowerCamelCase`. Prefer explicit domain names such as `ModelRegistry`, `ServeOptionParser`, or `OpenAITranslator`.

## Testing Guidelines

Tests use Swift Testing (`import Testing`) with `@Suite`, `@Test`, and `#expect`. Name test files after the unit or behavior, for example `ServeOptionParserTests.swift` or `ModelRegistryTests.swift`. Prefer isolated temporary files and clean them with `defer` when testing persistence. Run `make test` before submitting.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, capitalized subjects, for example `Fix tool calls blocked by ThinkingState filter` or `Add context_window/max_output_tokens to model info`.

Pull requests should include a concise description, linked issue when applicable, test results (`make test`, plus `make xcode-build` for runtime changes), and API examples only when behavior is user-visible.

## Security & Configuration Tips

Do not commit downloaded models, local caches, tokens, or generated build products. When changing server binding behavior, preserve the existing safety rule: non-loopback binds require `--auth-token`.
