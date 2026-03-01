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
        
        // --- ANCHORED PROMPT RECONSTRUCTION ---
        var systemPart = ""
        var discoveryAnchor = ""
        var recentUserPart = ""
        var lastObservation = ""
        
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemPart = String((sys.content ?? "").prefix(800))
        }
        
        for msg in chatReq.messages {
            if msg.role != "user" && msg.role != "system" {
                let content = msg.content ?? ""
                if content.contains("Tables_in") || content.contains("Field") {
                    discoveryAnchor = String(content.prefix(1000))
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
        
        SCHEMA:
        \(discoveryAnchor)
        
        LAST RESULT:
        \(lastObservation)
        
        GOAL:
        \(recentUserPart)
        
        CRITICAL: To use tools, you MUST use the format: [TOOL: name("args")]
        Example: [TOOL: mariadb_query("SELECT * FROM table")]
        Assistant:
        """
        
        let workerIndex = nextIndex()
        let worker = await pool.getWorker(for: workerIndex)
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // --- GREEDY MULTI-FORMAT TOOL PARSING ---
            var toolCalls: [OpenAIToolCall] = []
            let knownTools = ["mariadb_query", "mariadb_export", "read_data_chunk", "summarize_data", "compress_context", "read_file", "write_file", "run_shell", "think", "list_files", "search_files", "subtask", "execute"]
            
            // Pattern 1: [TOOL: name("args")] - Our preferred rigid format
            let pattern1 = "\\[TOOL:\\s*([a-zA-Z0-9_]+)\\s*\\((.*)\\)\\]"
            
            // Pattern 2: name("args") or name "args" - Model's natural tendency
            let pattern2 = "([a-zA-Z0-9_]+)\\s*(?:\\(|\\s+)([^\\n\\)]+)\\)?"
            
            let combinedPattern = "(\(pattern1))|(\(pattern2))"
            
            if let regex = try? NSRegularExpression(pattern: combinedPattern, options: [.dotMatchesLineSeparators]) {
                let nsRange = NSRange(trimmed.startIndex..., in: trimmed)
                let matches = regex.matches(in: trimmed, options: [], range: nsRange)
                
                for match in matches {
                    // Extract name and args based on which pattern matched
                    var name = ""
                    var args = ""
                    
                    if match.range(at: 2).location != NSNotFound {
                        // Pattern 1 matched
                        name = String(trimmed[Range(match.range(at: 2), in: trimmed)!]).trimmingCharacters(in: .whitespaces)
                        args = String(trimmed[Range(match.range(at: 3), in: trimmed)!]).trimmingCharacters(in: .whitespaces)
                    } else if match.range(at: 5).location != NSNotFound {
                        // Pattern 2 matched
                        name = String(trimmed[Range(match.range(at: 5), in: trimmed)!]).trimmingCharacters(in: .whitespaces)
                        args = String(trimmed[Range(match.range(at: 6), in: trimmed)!]).trimmingCharacters(in: .whitespaces)
                    }
                    
                    if knownTools.contains(name) {
                        // Cleanup args: convert simple strings to JSON
                        if !args.hasPrefix("{") {
                            let cleanStr = args.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                            if name == "mariadb_query" || name == "mariadb_export" {
                                args = "{\"query\": \"\(cleanStr)\"}"
                            } else if name == "read_file" || name == "list_files" || name == "search_files" {
                                args = "{\"path\": \"\(cleanStr)\", \"query\": \"\(cleanStr)\"}"
                            } else if name == "think" {
                                args = "{\"note\": \"\(cleanStr)\"}"
                            } else if name == "run_shell" {
                                args = "{\"command\": \"\(cleanStr)\"}"
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
            
            // Sample busy count BEFORE finishing
            let busyCount = await pool.busyWorkerCount()
            let finalBusy = max(busyCount, 1) // Always at least 1 since we are running
            
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
            throw Abort(.internalServerError, reason: "Apple Foundation Model error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String {
        struct SummarizeRequest: Content {
            let text: String
        }
        let summarizeReq = try req.content.decode(SummarizeRequest.self)
        let prompt = "Summarize concisely: \(summarizeReq.text.prefix(2500))\nSummary:"
        let worker = await pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
