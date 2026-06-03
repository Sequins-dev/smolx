import ArgumentParser
import Darwin
import Foundation
import Logging

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Run the HTTP server exposing OpenAI- and Anthropic-compatible endpoints."
    )

    @Option(name: .long, help: "Port to listen on.")
    var port: Int = 8080

    @Option(
        name: .long,
        help: "Address to bind. Use 0.0.0.0 to allow non-loopback access (requires --auth-token).")
    var bind: String = "127.0.0.1"

    @Option(
        name: .long,
        help: "Bearer token required on inbound requests. Mandatory when --bind is non-loopback.")
    var authToken: String?

    @Option(
        name: .long,
        help:
            "Evict loaded models whenever system available memory drops below this floor (e.g. '1GB', '512MB'). Default: 1GB."
    )
    var keepFree: String?

    @Option(
        name: .long,
        help:
            "Unload a model this long after the last active request releases it (e.g. '30s', '2m', '10m'). Default: 30s."
    )
    var idleTimeout: String?

    @Option(name: .long, help: "Cap on the number of models that can be resident at once.")
    var maxConcurrent: Int?

    @Flag(name: .shortAndLong, help: "Increase log verbosity (-v: debug, -vv: trace).")
    var verbose: Int

    func run() async throws {
        // Hard-kill on Ctrl+C / kill. We can't rely on a raw `signal(SIGINT,
        // ...)` handler because Hummingbird's `runService` calls
        // `signal(SIGINT, SIG_IGN)` via swift-service-lifecycle and replaces
        // ours with a DispatchSource — which then stalls when its queue is
        // busy (e.g. during MLX generation). The textbook fix is to block
        // SIGINT/SIGTERM in every thread and have one dedicated POSIX thread
        // `sigwait()` for them. The kernel routes blocked signals to whichever
        // thread is sigwaiting, so it always wins — even when Hummingbird's
        // dispatch queue is wedged. `_exit(0)` skips Swift cleanup, which is
        // fine: the OS reclaims MLX GPU memory on process death.
        installHardKillThread()

        let level: Logger.Level = verbose >= 2 ? .trace : (verbose >= 1 ? .debug : .info)
        LoggingSystem.bootstrap { label in
            var h = StreamLogHandler.standardOutput(label: label)
            h.logLevel = level
            return h
        }

        let logger = Logger(label: "smolx")
        let registry = ModelRegistry()
        CodexCatalog.write((try? registry.load()) ?? [], logger: logger)

        var settings = ModelManager.Settings.default
        if let s = keepFree, let bytes = SystemMemory.parse(s) {
            settings.keepFreeBytes = bytes
        }
        if let raw = idleTimeout, let secs = SystemMemory.parseDuration(raw) {
            settings.idleTimeout = secs
        }
        settings.maxConcurrent = maxConcurrent

        let manager = ModelManager(
            registry: registry,
            factory: MLXProviderFactory(),
            settings: settings,
            logger: logger)
        await manager.start()
        defer { Task { await manager.stop() } }

        try await HTTPServer.run(
            config: .init(host: bind, port: port, authToken: authToken),
            manager: manager,
            registry: registry,
            logger: logger)
    }
}

/// Block SIGINT/SIGTERM in the calling thread (and all threads spawned
/// after — `pthread_sigmask` is inherited) and spawn a detached pthread
/// that `sigwait()`s for them. On signal it writes a one-line shutdown
/// notice to stderr (async-signal-safe `write`, not `print`) and calls
/// `_exit(0)`. This sidesteps Hummingbird/swift-service-lifecycle's
/// DispatchSource signal handling entirely — the kernel delivers blocked
/// signals to whatever thread is sigwaiting, so dispatch-queue stalls
/// can no longer keep the process alive.
private func installHardKillThread() {
    var set = sigset_t()
    sigemptyset(&set)
    sigaddset(&set, SIGINT)
    sigaddset(&set, SIGTERM)
    pthread_sigmask(SIG_BLOCK, &set, nil)

    // pthread_create takes a single `void*` context. Allocate the signal
    // set on the heap and hand the pointer to the thread; the thread owns
    // the allocation for its lifetime (which ends at _exit, so we don't
    // bother freeing).
    let pset = UnsafeMutablePointer<sigset_t>.allocate(capacity: 1)
    pset.initialize(to: set)

    var tid = pthread_t(bitPattern: 0)
    let rc = pthread_create(
        &tid, nil,
        { ctx in
            let set = ctx.assumingMemoryBound(to: sigset_t.self)
            var caught: Int32 = 0
            sigwait(set, &caught)
            let msg = "\nsmolx: shutting down on signal \(caught)\n"
            _ = msg.withCString { write(STDERR_FILENO, $0, strlen($0)) }
            Darwin._exit(0)
        }, pset)
    if rc == 0, let tid {
        pthread_detach(tid)
    }
}
