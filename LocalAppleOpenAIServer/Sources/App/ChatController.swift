import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    let id: Int
    private(set) var isBusy: Bool = false
    
    init(id: Int) {
        self.id = id
    }
    
    func respond(to prompt: String) async throws -> String {
        self.isBusy = true
        defer { self.isBusy = false }
        
        print("Worker [\(id)]: Thinking (High Priority)...")
        fflush(stdout)
        
        let generatedContent = try await Task.detached(priority: .userInitiated) {
            let session = LanguageModelSession(model: .default)
            let response = try await session.respond(to: prompt)
            return response.content
        }.value
        
        print("Worker [\(id)]: Done.")
        fflush(stdout)
        return generatedContent
    }
}

@available(macOS 26.0, *)
actor ModelPool {
    private var workerDict: [Int: ModelWorker] = [:]
    private let maxWorkers: Int
    
    init() {
        self.maxWorkers = 120
        for i in 0..<20 {
            workerDict[i] = ModelWorker(id: i)
        }
    }
    
    func getWorker(for index: Int) async -> ModelWorker {
        let workerID = index % maxWorkers
        if let existing = workerDict[workerID] { return existing }
        let newWorker = ModelWorker(id: workerID)
        workerDict[workerID] = newWorker
        return newWorker
    }

    func busyWorkerCount() async -> Int {
        var count = 0
        for worker in workerDict.values {
            if await worker.isBusy { count += 1 }
        }
        return count
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
        
        // --- DIET PROMPT RECONSTRUCTION ---
        var systemPart = ""
        var discoveryAnchor = ""
        var recentUserPart = ""
        var lastObservation = ""
        
        // 1. System Prompt (Reduced)
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemPart = String((sys.content ?? "").prefix(500))
        }
        
        // 2. Find ONLY the best schema anchor
        for msg in chatReq.messages.reversed() { // Look at newest first
            let content = msg.content ?? ""
            if content.contains("Field") && content.contains("Type") {
                // It's a DESCRIBE result, heavily truncate it to save tokens
                discoveryAnchor = "SCHEMA:\n" + String(content.prefix(800))
                break
            } else if content.contains("Tables_in") && discoveryAnchor.isEmpty {
                // Fallback to SHOW TABLES
                discoveryAnchor = "TABLES:\n" + String(content.prefix(500))
            }
        }
        
        // 3. User Goal
        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) {
            recentUserPart = String((lastUser.content ?? "").prefix(400))
        }
        
        // 4. Last Tool Result
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) {
            lastObservation = String((lastObs.content ?? "").prefix(800))
        }
        
        let prompt = """
        \(systemPart)
        
        \(discoveryAnchor)
        
        LAST RESULT:
        \(lastObservation)
        
        GOAL:
        \(recentUserPart)
        
        ACTION FORMAT: [TOOL: name("arg1")]
        """
        
        let workerIndex = nextIndex()
        let worker = await pool.getWorker(for: workerIndex)
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let busyCount = await pool.busyWorkerCount()
            let finalBusy = max(busyCount, 1)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // --- SURGICAL TOOL PARSING ---
            var toolCalls: [OpenAIToolCall] = []
            
            // This regex specifically targets content INSIDE double quotes
            // e.g. [TOOL: mariadb_query("SELECT * FROM table")] -> name: mariadb_query, arg: SELECT * FROM table
            let pattern = "\\[TOOL:\\s*([a-zA-Z0-9_]+)\\(\"([^\"]+)\"(?:,\\s*\"([^\"]+)\")?\\)\\]"
            
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let nsRange = NSRange(trimmed.startIndex..., in: trimmed)
                let matches = regex.matches(in: trimmed, options: [], range: nsRange)
                
                for match in matches {
                    if let nameRange = Range(match.range(at: 1), in: trimmed),
                       let arg1Range = Range(match.range(at: 2), in: trimmed) {
                        
                        let name = String(trimmed[nameRange])
                        let arg1 = String(trimmed[arg1Range])
                        
                        var arg2: String? = nil
                        if match.range(at: 3).location != NSNotFound, let arg2Range = Range(match.range(at: 3), in: trimmed) {
                            arg2 = String(trimmed[arg2Range])
                        }
                        
                        var dict: [String: String] = [:]
                        if name == "mariadb_query" || name == "mariadb_export" {
                            dict["query"] = arg1
                        } else if name == "mariadb_search" {
                            dict["table"] = arg1
                            if let a2 = arg2 { dict["query"] = a2 }
                        } else if name == "mariadb_sample" {
                            dict["table"] = arg1
                        } else if name == "think" {
                            dict["note"] = arg1
                        } else if name == "read_file" || name == "list_files" {
                            dict["path"] = arg1
                        }
                        
                        if let data = try? JSONSerialization.data(withJSONObject: dict),
                           let finalArgsStr = String(data: data, encoding: .utf8) {
                            toolCalls.append(OpenAIToolCall(
                                id: "call_" + UUID().uuidString.prefix(8),
                                type: "function",
                                function: OpenAIToolCallFunction(name: name, arguments: finalArgsStr)
                            ))
                        }
                    }
                }
            }
            
            let responseObj = OpenAIChatResponse(
                id: "chatcmpl-" + UUID().uuidString,
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: chatReq.model,
                choices: [OpenAIChoice(
                    index: 0,
                    message: OpenAIMessageResponse(
                        role: "assistant",
                        content: toolCalls.isEmpty ? trimmed : nil,
                        tool_calls: toolCalls.isEmpty ? nil : toolCalls
                    ),
                    finish_reason: toolCalls.isEmpty ? "stop" : "tool_calls"
                )],
                usage: OpenAIUsage(
                    prompt_tokens: prompt.count / 4,
                    completion_tokens: trimmed.count / 4,
                    total_tokens: (prompt.count + trimmed.count) / 4
                )
            )
            
            var rawRes = try JSONEncoder().encode(responseObj)
            if var json = try JSONSerialization.jsonObject(with: rawRes) as? [String: Any],
               var usage = json["usage"] as? [String: Any] {
                usage["active_workers"] = finalBusy
                usage["worker_id"] = workerIndex % 120
                json["usage"] = usage
                rawRes = try JSONSerialization.data(withJSONObject: json)
            }
            
            let res = Response(status: .ok, body: .init(data: rawRes))
            res.headers.replaceOrAdd(name: .contentType, value: "application/json")
            return res
            
        } catch {
            print("ChatController: ERROR - \(error)")
            throw Abort(.internalServerError, reason: "Apple Bridge Error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String {
        struct SummarizeRequest: Content {
            let text: String
        }
        let summarizeReq = try req.content.decode(SummarizeRequest.self)
        let prompt = "Summarize concisely: \(summarizeReq.text.prefix(2000))\nSummary:"
        let worker = await pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
