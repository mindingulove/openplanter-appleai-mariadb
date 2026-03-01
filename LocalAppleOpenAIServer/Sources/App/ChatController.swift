import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    private let id: Int
    
    init(id: Int) {
        self.id = id
    }
    
    func respond(to prompt: String) async throws -> String {
        print("Worker [\(id)]: Incoming request (\(prompt.count) chars)...")
        fflush(stdout)
        
        // Every worker creates a fresh, isolated session
        let session = LanguageModelSession(model: .default)
        
        print("Worker [\(id)]: Starting generation...")
        fflush(stdout)
        
        let response = try await session.respond(to: prompt)
        
        print("Worker [\(id)]: Done. Output size: \(response.content.count) chars")
        fflush(stdout)
        return response.content
    }
}

@available(macOS 26.0, *)
final class ModelPool: Sendable {
    private let workers: [ModelWorker]
    
    init() {
        let totalRAM = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let cpuCores = ProcessInfo.processInfo.processorCount
        let safeModelRAM = max(0, Int(totalRAM) - 8)
        let workerCount = min(max(1, safeModelRAM / 4), cpuCores)
        
        print("💻 Mac Specs: \(totalRAM)GB RAM, \(cpuCores) Cores")
        print("🧠 Model Pool: Spawning \(workerCount) workers...")
        fflush(stdout)
        
        self.workers = (0..<workerCount).map { ModelWorker(id: $0) }
    }
    
    func getWorker(for index: Int) -> ModelWorker {
        return workers[index % workers.count]
    }
}

@available(macOS 26.0, *)
final class ChatController: @unchecked Sendable {
    private let pool = ModelPool()
    private var counter: Int = 0
    private let lock = NSLock()

    init() {
        print("✅ Apple Foundation Model integration INITIALIZED with Resource-Aware Pooling.")
        fflush(stdout)
    }
    
    private func nextIndex() -> Int {
        lock.lock()
        defer { lock.unlock() }
        counter += 1
        return counter
    }
    
    @Sendable
    func createChatCompletion(req: Request) async throws -> Response {
        let chatReq = try req.content.decode(OpenAIChatRequest.self)
        
        // --- ULTRA-AGGRESSIVE TRUNCATION ---
        // Local Apple models (4k window) are very sensitive.
        // We only take the LATEST message and cap it at 1000 chars (~250-300 tokens).
        let userMessage = chatReq.messages.last?.content ?? "Hello"
        let cappedContent = userMessage.count > 1000 ? String(userMessage.prefix(1000)) + "..." : userMessage
        
        let prompt = """
        Instruction: You are OpenPlanter. Use tools.
        - `mariadb_query`: {"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "mariadb_query", "arguments": "{\\"query\\": \\"SHOW TABLES;\\"}"}}]}
        
        User: \(cappedContent)
        Assistant:
        """
        
        // Round-robin worker selection
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let finalOutput = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            var toolCalls: [[String: Any]]? = nil
            if finalOutput.contains("mariadb_query") {
                var dbQuery = "SHOW TABLES;"
                if let range = finalOutput.range(of: "SELECT", options: .caseInsensitive) {
                    dbQuery = String(finalOutput[range.lowerBound...]).components(separatedBy: "\"")[0].components(separatedBy: "}")[0]
                }
                toolCalls = [[
                    "id": "call_\(UUID().uuidString.prefix(8))",
                    "type": "function",
                    "function": ["name": "mariadb_query", "arguments": "{\"query\": \"\(dbQuery)\"}"]
                ]]
            }
            
            let responseObj: [String: Any] = [
                "id": "chatcmpl-\(UUID().uuidString)",
                "object": chatReq.stream == true ? "chat.completion.chunk" : "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": chatReq.model,
                "choices": [[
                    "index": 0,
                    (chatReq.stream == true ? "delta" : "message"): [
                        "role": "assistant",
                        "content": toolCalls == nil ? finalOutput : nil,
                        "tool_calls": toolCalls
                    ],
                    "finish_reason": toolCalls == nil ? "stop" : "tool_calls"
                ]],
                "usage": [
                    "prompt_tokens": prompt.count / 4,
                    "completion_tokens": finalOutput.count / 4,
                    "total_tokens": (prompt.count + finalOutput.count) / 4
                ]
            ]
            
            if chatReq.stream == true {
                let sseData = "data: \(try String(data: JSONSerialization.data(withJSONObject: responseObj), encoding: .utf8)!)\n\ndata: [DONE]\n\n"
                let res = try await sseData.encodeResponse(for: req)
                res.headers.replaceOrAdd(name: .contentType, value: "text/event-stream")
                return res
            } else {
                let data = try JSONSerialization.data(withJSONObject: responseObj)
                let res = Response(status: .ok, body: .init(data: data))
                res.headers.replaceOrAdd(name: .contentType, value: "application/json")
                return res
            }
            
        } catch {
            print("ChatController: ERROR - \(error)")
            fflush(stdout)
            throw Abort(.internalServerError, reason: "Apple Foundation Model error: \(error.localizedDescription)")
        }
    }
}
