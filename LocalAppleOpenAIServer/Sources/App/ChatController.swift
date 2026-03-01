import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    let id: Int
    private let session: LanguageModelSession
    
    init(id: Int) {
        self.id = id
        self.session = LanguageModelSession(model: .default)
    }
    
    func respond(to prompt: String) async throws -> String {
        print("Worker [\(id)]: Thinking (High Priority)...")
        fflush(stdout)
        
        let generatedContent = try await Task.detached(priority: .userInitiated) {
            let response = try await self.session.respond(to: prompt)
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
        if let envWorkers = ProcessInfo.processInfo.environment["APPLE_BRIDGE_WORKERS"], let count = Int(envWorkers) {
            self.maxWorkers = count
        } else {
            self.maxWorkers = 120
        }
        
        print("🧠 Model Pool: Dynamic mode enabled (Cap: \(maxWorkers) workers)")
        
        print("🔥 Pre-warming 20 workers...")
        for i in 0..<20 {
            workerDict[i] = ModelWorker(id: i)
        }
        fflush(stdout)
    }
    
    func getWorker(for index: Int) async -> ModelWorker {
        let workerID = index % maxWorkers
        if let existing = workerDict[workerID] { return existing }
        let newWorker = ModelWorker(id: workerID)
        workerDict[workerID] = newWorker
        return newWorker
    }

    func activeWorkerCount() -> Int {
        return workerDict.count
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
        
        // --- ANCHORED PROMPT RECONSTRUCTION ---
        var systemPart = ""
        var toolsPart = ""
        var discoveryAnchor = ""
        var recentUserPart = ""
        var lastObservation = ""
        
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemPart = String((sys.content ?? "").prefix(800))
        }
        
        if let tools = chatReq.tools {
            toolsPart = "Available Tools:\n"
            for t in tools.prefix(12) { 
                let params = t.function.parameters?.stringify() ?? "{}"
                toolsPart += "- \(t.function.name)(\(params))\n"
            }
        }
        
        for msg in chatReq.messages {
            if msg.role != "user" && msg.role != "system" {
                let content = msg.content ?? ""
                if content.contains("Tables_in") || content.contains("Field") || content.contains("Type") {
                    discoveryAnchor = String(content.prefix(1200))
                    break
                }
            }
        }
        
        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) {
            recentUserPart = String((lastUser.content ?? "").prefix(600))
        }
        
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) {
            lastObservation = String((lastObs.content ?? "").prefix(1200))
        }
        
        let prompt = """
        \(systemPart)
        
        \(toolsPart)
        
        SCHEMA:
        \(discoveryAnchor)
        
        LAST RESULT:
        \(lastObservation)
        
        GOAL:
        \(recentUserPart)
        
        ACT NOW. Use tool calls like: mariadb_query("SELECT...")
        Assistant:
        """
        
        let workerIndex = nextIndex()
        let worker = await pool.getWorker(for: workerIndex)
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // --- ULTRA-ROBUST TOOL CALL PARSING ---
            var toolCalls: [OpenAIToolCall] = []
            
            // 1. Try to find JSON arrays first (the model likes this)
            if trimmed.contains("[") && trimmed.contains("]") {
                let startIdx = trimmed.firstIndex(of: "[")!
                let endIdx = trimmed.lastIndex(of: "]")!
                let jsonPart = String(trimmed[startIdx...endIdx])
                
                if let data = jsonPart.data(using: .utf8),
                   let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    for item in array {
                        let name = (item["tool"] as? String) ?? (item["name"] as? String) ?? ""
                        if !name.isEmpty {
                            var argsStr = "{}"
                            if let argsObj = item["args"] ?? item["parameters"] ?? item["arguments"] {
                                if let argsData = try? JSONSerialization.data(withJSONObject: argsObj),
                                   let s = String(data: argsData, encoding: .utf8) {
                                    argsStr = s
                                }
                            }
                            toolCalls.append(OpenAIToolCall(
                                id: "call_" + UUID().uuidString.prefix(8),
                                type: "function",
                                function: OpenAIToolCallFunction(name: name, arguments: argsStr)
                            ))
                        }
                    }
                }
            }
            
            // 2. Fallback to Regex for name("args") format
            if toolCalls.isEmpty {
                let pattern = "([a-zA-Z0-9_]+)\\s*\\((.*)\\)"
                if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                    let nsRange = NSRange(trimmed.startIndex..., in: trimmed)
                    let matches = regex.matches(in: trimmed, options: [], range: nsRange)
                    
                    for match in matches {
                        if let nameRange = Range(match.range(at: 1), in: trimmed),
                           let argsRange = Range(match.range(at: 2), in: trimmed) {
                            let name = String(trimmed[nameRange]).trimmingCharacters(in: .whitespaces)
                            var args = String(trimmed[argsRange]).trimmingCharacters(in: .whitespaces)
                            
                            // Handle simple string args: name("val") -> {"query": "val"}
                            if args.hasPrefix("\"") && args.hasSuffix("\"") && !args.contains(":") {
                                if name == "mariadb_query" || name == "mariadb_export" {
                                    args = "{\"query\": \(args)}"
                                } else if name == "read_file" || name == "list_files" {
                                    args = "{\"path\": \(args)}"
                                }
                            }
                            
                            toolCalls.append(OpenAIToolCall(
                                id: "call_" + UUID().uuidString.prefix(8),
                                type: "function",
                                function: OpenAIToolCallFunction(name: name, arguments: args)
                            ))
                        }
                    }
                }
            }
            
            let activeCount = await pool.activeWorkerCount()
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
                usage["active_workers"] = activeCount
                usage["worker_id"] = workerIndex % 120
                json["usage"] = usage
                rawRes = try JSONSerialization.data(withJSONObject: json)
            }
            
            let res = Response(status: .ok, body: .init(data: rawRes))
            res.headers.replaceOrAdd(name: .contentType, value: "application/json")
            return res
            
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
        let prompt = "Squeeze this data into one core fact sentence: \(summarizeReq.text.prefix(2500))\nFact:"
        let worker = await pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
