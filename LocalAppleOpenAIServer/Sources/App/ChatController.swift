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
        print("Worker [\(id)]: Generation start (\(prompt.count) chars)...")
        fflush(stdout)
        
        let session = LanguageModelSession(model: .default)
        let response = try await session.respond(to: prompt)
        
        print("Worker [\(id)]: Generation end.")
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
        let workerCount = min(Int(totalRAM / 2), cpuCores)
        
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
        print("✅ Apple Foundation Model integration INITIALIZED.")
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
        
        // --- PROMPT RECONSTRUCTION (Ultra-Safe) ---
        // We strictly limit the history to avoid the 4091 token crash.
        var prompt = ""
        
        // 1. System Prompt (The Rules)
        if let systemMsg = chatReq.messages.first(where: { $0.role == "system" }) {
            prompt += "Instruction: \(String((systemMsg.content ?? "").prefix(800)))\n"
        }
        
        // 2. Goal (The original question)
        if let firstUserMsg = chatReq.messages.first(where: { $0.role == "user" }) {
            prompt += "Goal: \(String((firstUserMsg.content ?? "").prefix(400)))\n"
        }
        
        // 3. Current Observation (The last result)
        if let lastMsg = chatReq.messages.last, lastMsg.role != "user" {
            prompt += "Observation: \(String((lastMsg.content ?? "").prefix(1200)))\n"
        }
        
        // 4. Force Nudge
        prompt += """
        
        CRITICAL: If the 'Observation' above already answers the 'Goal', provide the final answer to the user now. 
        If you need more info from MariaDB, use `mariadb_query`.
        Assistant:
        """
        
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let finalOutput = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Repair SQL Hallucinations
            var toolCalls: [[String: Any]]? = nil
            if finalOutput.contains("mariadb_query") || finalOutput.contains("SHOW TABLES") || (finalOutput.contains("SELECT") && finalOutput.contains("FROM")) {
                var dbQuery = "SHOW TABLES;"
                if let range = finalOutput.range(of: "SELECT", options: .caseInsensitive) {
                    let part = String(finalOutput[range.lowerBound...])
                    dbQuery = part.components(separatedBy: "\n")[0].components(separatedBy: ";")[0] + ";"
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
        let prompt = "Instruction: Summarize this briefly (max 300 chars): \(summarizeReq.text.prefix(2000))\nSummary:"
        let worker = pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
