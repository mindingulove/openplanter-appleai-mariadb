import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    private let id: Int
    private let session: LanguageModelSession
    
    init(id: Int) {
        self.id = id
        self.session = LanguageModelSession(model: .default)
    }
    
    func respond(to prompt: String) async throws -> String {
        print("Worker [\(id)]: Start (Prompt: \(prompt.count) chars)")
        fflush(stdout)
        
        let response = try await session.respond(to: prompt)
        
        print("Worker [\(id)]: Done.")
        fflush(stdout)
        return response.content
    }
}

@available(macOS 26.0, *)
final class ModelPool: Sendable {
    private let workers: [ModelWorker]
    
    init() {
        // --- SANE SCALING FOR 16GB RAM ---
        // Spawning 4 workers is the "Sweet Spot" for M1 16GB.
        // It provides high concurrency without risking system freezes.
        let workerCount = 4
        
        print("🧠 Optimized Pool: Spawning \(workerCount) parallel AI workers...")
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
        print("✅ Apple Silicon Optimized Bridge: ACTIVE (4 Workers)")
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
        
        // --- PROMPT RECONSTRUCTION ---
        var prompt = ""
        
        // 1. System Prompt (Rules)
        if let systemMsg = chatReq.messages.first(where: { $0.role == "system" }) {
            prompt += "Instruction: \(String((systemMsg.content ?? "").prefix(1000)))\n"
        }
        
        // 2. Goal (Original Question)
        if let firstUser = chatReq.messages.first(where: { $0.role == "user" }) {
            prompt += "Goal: \(String((firstUser.content ?? "").prefix(1000)))\n"
        }
        
        // 3. Current Observation (The data)
        if let last = chatReq.messages.last, last.role != "user" {
            // Give 3000 chars to the last observation (the data)
            prompt += "Observation: \(String((last.content ?? "").prefix(3000)))\n"
        }
        
        prompt += "\nAssistant:"
        
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let finalOutput = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Reparation logic for tool calls
            var toolCalls: [[String: Any]]? = nil
            if finalOutput.contains("mariadb_query") || finalOutput.contains("SELECT") || finalOutput.contains("SHOW TABLES") {
                var dbQuery = "SHOW TABLES;"
                if let range = finalOutput.range(of: "SELECT", options: .caseInsensitive) {
                    let part = String(finalOutput[range.lowerBound...])
                    dbQuery = part.components(separatedBy: "\n")[0].components(separatedBy: "`")[0].components(separatedBy: "\"")[0].components(separatedBy: ";")[0] + ";"
                } else if let range = finalOutput.range(of: "DESCRIBE", options: .caseInsensitive) {
                    let part = String(finalOutput[range.lowerBound...])
                    dbQuery = part.components(separatedBy: "\n")[0].components(separatedBy: "`")[0].components(separatedBy: "\"")[0].components(separatedBy: ";")[0] + ";"
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
            throw Abort(.internalServerError, reason: "Apple Foundation Model error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String {
        struct SummarizeRequest: Content {
            let text: String
        }
        let summarizeReq = try req.content.decode(SummarizeRequest.self)
        let prompt = "Summarize the following technical findings briefly (max 300 chars): \(summarizeReq.text.prefix(3000))"
        let worker = pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
