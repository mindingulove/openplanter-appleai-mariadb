import Vapor
import NIOPosix

struct PortFileLifecycle: LifecycleHandler {
    func didBoot(_ app: Application) throws {
        Task {
            for _ in 0..<20 {
                if let address = app.http.server.shared.localAddress, let port = address.port {
                    let portFilePath = "/tmp/openplanter_bridge_port"
                    do {
                        try String(port).write(toFile: portFilePath, atomically: true, encoding: .utf8)
                        app.logger.info("🚀 Bridge port \(port) saved to \(portFilePath)")
                        return
                    } catch {
                        app.logger.error("❌ Failed to write port file: \(error)")
                    }
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
    
    func shutdown(_ app: Application) {
        try? FileManager.default.removeItem(atPath: "/tmp/openplanter_bridge_port")
    }
}

public func configure(_ app: Application) throws {
    // Increase maximum body size for large prompts
    app.routes.defaultMaxBodySize = "10mb"
    
    // Performance Tuning: Use maximum available threads for the server
    let cores = ProcessInfo.processInfo.activeProcessorCount
    app.logger.info("⚡️ Performance Tuning: Using \(cores) cores for event loops")
    
    // Configure port: check environment variable PORT, otherwise use 0 (auto-assign free port)
    if let portString = ProcessInfo.processInfo.environment["PORT"], let port = Int(portString) {
        app.http.server.configuration.port = port
    } else {
        app.http.server.configuration.port = 0
    }
    
    // High-concurrency settings
    app.http.server.configuration.supportPipelining = true
    app.http.server.configuration.requestDecompression = .enabled
    app.http.server.configuration.responseCompression = .enabled
    
    // Register the port-saving logic
    app.lifecycle.use(PortFileLifecycle())
    
    // Register routes
    try routes(app)
}
