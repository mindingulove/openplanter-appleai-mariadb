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
        print("Worker [\(id)]: Active (Prompt: \(prompt.count) chars)")
        fflush(stdout)
        
        let session = LanguageModelSession(model: .default)
        let response = try await session.respond(to: prompt)
        
        print("Worker [\(id)]: Generation complete.")
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
        print("🧠 Model Pool: \(workerCount) workers ready.")
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
        print("✅ Apple Foundation Model integration INITIALIZED with Contextual Linking.")
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
        
        // --- INTELLIGENT PROMPT RECONSTRUCTION ---
        // Instead of aggressive truncation, we build a structured "Thought Context"
        // 4091 tokens ≈ 16,000 characters. We'll be conservative and aim for 8,000 chars total.
        
        var systemRules = ""
        var originalGoal = ""
        var latestObservation = ""
        var middleContext = ""
        
        // 1. Identify critical components
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemRules = String((sys.content ?? "").prefix(1500))
        }
        
        if let firstUser = chatReq.messages.first(where: { $0.role == "user" }) {
            originalGoal = String((firstUser.content ?? "").prefix(1000))
        }
        
        if let lastMsg = chatReq.messages.last, lastMsg.role != "user" {
            latestObservation = String((lastMsg.content ?? "").prefix(3000))
        }
        
        // 2. Fill the gap with recent history (if space permits)
        let remainingMessages = chatReq.messages.dropFirst().dropLast()
        if !remainingMessages.isEmpty {
            let recent = remainingMessages.suffix(3)
            for m in recent {
                middleContext += "\(m.role.capitalized): \(String((m.content ?? "").prefix(500)))\n"
            }
        }
        
        // 3. Assemble the Structured Prompt
        let prompt = """
        Instruction: \(systemRules)
        
        Goal: \(originalGoal)
        
        Recent Context:
        \(middleContext)
        
        Current Observation:
        \(latestObservation)
        
        CRITICAL INSTRUCTION:
        If the 'Current Observation' contains the answer to the 'Goal', provide the final answer to the user now.
        If you need more information (e.g. table details or query results), use the `mariadb_query` tool.
        Do NOT repeat the same query if the result is already in the 'Recent Context' or 'Observation'.
        
        Assistant:
        """
        
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let finalOutput = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Reparation logic for tool calls (if model wrote markdown instead of JSON)
            var toolCalls: [[String: Any]]? = nil
            if finalOutput.contains("mariadb_query") || finalOutput.contains("SELECT") {
                var dbQuery = "SHOW TABLES;"
                if let range = finalOutput.range(of: "SELECT", options: .caseInsensitive) {
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
        let prompt = "Instruction: Summarize this briefly (max 300 chars): \(summarizeReq.text.prefix(2000))\nSummary:"
        let worker = pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
