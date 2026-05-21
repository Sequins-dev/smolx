import Foundation

/// Direct URLSession downloader that streams bytes into a `<dest>.partial`
/// file and updates a `Foundation.Progress` as bytes arrive. We do this
/// ourselves because `URLSession.download(for:delegate:)` on macOS does not
/// reliably call the task-specific `URLSessionDownloadDelegate.didWriteData`
/// callback — the Progress object swift-huggingface passes to its delegate
/// stays at 0 throughout the transfer and only the final completion is
/// observable. A session-level `URLSessionDataDelegate` IS called for every
/// chunk, which is what we use here.
///
/// Side benefit: because we control the partial file ourselves, **byte-level
/// resume across crashes works**. A killed process leaves `<dest>.partial`
/// intact; the next call to `download(...)` for the same destination sends
/// `Range: bytes=<existing-size>-` and appends to the file.
final class StreamingDownloader: @unchecked Sendable {

    static let shared = StreamingDownloader()

    private let session: URLSession
    private let delegate: Delegate

    init() {
        let config = URLSessionConfiguration.default
        // 60 s to first byte; up to 7 days for the whole transfer so massive
        // shards don't hit a resource-timeout while still making progress.
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 7 * 24 * 3600
        config.httpMaximumConnectionsPerHost = 8

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1   // serialize delegate callbacks
        queue.name = "mlx-serve.streaming-downloader.delegate"

        let delegate = Delegate()
        self.delegate = delegate
        self.session = URLSession(
            configuration: config, delegate: delegate, delegateQueue: queue)
    }

    /// Downloads `url` to `destination`, appending to any existing `.partial`.
    /// `progress.completedUnitCount` is updated continuously; `totalUnitCount`
    /// is set from the response's Content-Length (resumed downloads include
    /// the already-on-disk portion in the total).
    func download(
        url: URL,
        destination: URL,
        progress: Foundation.Progress,
        headers: [String: String] = [:]
    ) async throws {
        let partial = destination.appendingPathExtension("partial")

        // Make sure the parent directory exists. The first `.partial` write
        // would fail otherwise.
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        let resumeFrom = Self.partialSize(at: partial)
        var request = URLRequest(url: url)
        for (k, v) in headers { request.setValue(v, forHTTPHeaderField: k) }
        if resumeFrom > 0 {
            request.setValue("bytes=\(resumeFrom)-", forHTTPHeaderField: "Range")
        }

        // Open `.partial` for append-write. We seek to end on open so even if
        // we made a mistake on resumeFrom the file isn't truncated.
        if !FileManager.default.fileExists(atPath: partial.path) {
            FileManager.default.createFile(atPath: partial.path, contents: nil)
        }
        let fileHandle = try FileHandle(forWritingTo: partial)
        try fileHandle.seekToEnd()

        // Initialise progress so the bar opens at the resumed offset.
        if resumeFrom > 0 {
            progress.completedUnitCount = resumeFrom
        }

        let response: HTTPURLResponse
        do {
            response = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation {
                    (cont: CheckedContinuation<HTTPURLResponse, Error>) in
                    let task = session.dataTask(with: request)
                    delegate.register(
                        taskID: task.taskIdentifier,
                        fileHandle: fileHandle,
                        progress: progress,
                        resumeFrom: resumeFrom,
                        continuation: cont)
                    task.resume()
                }
            } onCancel: {
                // We can't reach the task object cleanly from here, so we mark
                // all in-flight tasks for the delegate to cancel. Best-effort.
                delegate.cancelAll()
            }
        } catch {
            try? fileHandle.close()
            // Leave `.partial` on disk for the next attempt — that's the
            // whole point of resumability.
            throw error
        }
        try? fileHandle.close()

        guard (200..<300).contains(response.statusCode) else {
            // Server refused, e.g. 416 Range-Not-Satisfiable means our partial
            // is bigger than the server's file; discard it and let the caller
            // retry from scratch.
            if response.statusCode == 416 {
                try? FileManager.default.removeItem(at: partial)
            }
            throw URLError(.init(rawValue: response.statusCode))
        }

        // If we sent a Range header but the server returned 200 (full body),
        // it ignored our resume request and we've appended a full copy onto
        // existing partial bytes. Discard and start over next time.
        if resumeFrom > 0, response.statusCode == 200 {
            try? FileManager.default.removeItem(at: partial)
            throw URLError(.cannotParseResponse)  // transient ⇒ RetryPolicy retries from byte 0
        }

        // Atomic rename: any reader who sees `destination` exists can trust it.
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
    }

    /// Existing bytes on disk for resume; 0 if no partial exists.
    private static func partialSize(at url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) else {
            return 0
        }
        return (attrs[.size] as? Int64) ?? Int64(attrs[.size] as? Int ?? 0)
    }

    // MARK: - Session-level delegate

    /// Holds one entry per active dataTask. URLSession calls our delegate
    /// methods on the operation queue we configured — we serialize all access
    /// to `pending` with a lock since the OperationQueue is concurrent-safe
    /// but Swift's strict concurrency requires explicit protection.
    final class Delegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {

        private struct Pending {
            let fileHandle: FileHandle
            let progress: Foundation.Progress
            let continuation: CheckedContinuation<HTTPURLResponse, Error>
            var totalBytes: Int64
            var receivedResponse: HTTPURLResponse?
        }

        private var pending: [Int: Pending] = [:]
        private let lock = NSLock()

        func register(
            taskID: Int,
            fileHandle: FileHandle,
            progress: Foundation.Progress,
            resumeFrom: Int64,
            continuation: CheckedContinuation<HTTPURLResponse, Error>
        ) {
            lock.lock(); defer { lock.unlock() }
            pending[taskID] = Pending(
                fileHandle: fileHandle,
                progress: progress,
                continuation: continuation,
                totalBytes: resumeFrom,
                receivedResponse: nil)
        }

        func cancelAll() {
            // Best-effort cancellation hook called from
            // `withTaskCancellationHandler`. We don't hold task references
            // here (the dataTask is created and resumed inside the
            // continuation closure), so this is a no-op today — real
            // cancellation flows through Task.isCancelled checks in
            // the downloader, which short-circuits the next call attempt.
        }

        // MARK: URLSessionDataDelegate

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
        ) {
            lock.lock()
            if var p = pending[dataTask.taskIdentifier],
               let http = response as? HTTPURLResponse
            {
                p.receivedResponse = http
                let contentLength = http.expectedContentLength
                if contentLength > 0 {
                    p.progress.totalUnitCount = p.totalBytes + contentLength
                }
                pending[dataTask.taskIdentifier] = p
            }
            lock.unlock()
            completionHandler(.allow)
        }

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive data: Data
        ) {
            lock.lock()
            guard var p = pending[dataTask.taskIdentifier] else {
                lock.unlock(); return
            }
            do {
                try p.fileHandle.write(contentsOf: data)
                p.totalBytes += Int64(data.count)
                p.progress.completedUnitCount = p.totalBytes
                pending[dataTask.taskIdentifier] = p
                lock.unlock()
            } catch {
                lock.unlock()
                dataTask.cancel()
            }
        }

        func urlSession(
            _ session: URLSession,
            task: URLSessionTask,
            didCompleteWithError error: Error?
        ) {
            lock.lock()
            let pulled = pending.removeValue(forKey: task.taskIdentifier)
            lock.unlock()
            guard let p = pulled else { return }
            if let error {
                p.continuation.resume(throwing: error)
                return
            }
            if let received = p.receivedResponse {
                p.continuation.resume(returning: received)
            } else if let http = task.response as? HTTPURLResponse {
                p.continuation.resume(returning: http)
            } else {
                p.continuation.resume(throwing: URLError(.badServerResponse))
            }
        }
    }
}
