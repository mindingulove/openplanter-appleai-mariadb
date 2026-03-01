import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    let id: Int
    private(set) var isBusy: Bool = false
    init(id: Int) { self.id = id }
    func respond(to prompt: String) async throws -> String {
        self.isBusy = true
        defer { self.isBusy = false }
        print("Worker [\(id)]: Thinking (High Priority)...")
        fflush(stdout)
        let generatedContent = try await Task.detached(priority: .userInitiated) {
            let session = LanguageModelSession(model: .default)
            let response = try await session.respond(to: String(prompt.prefix(10000)))
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
    init() { for i in 0..<20 { workerDict[i] = ModelWorker(id: i) } }
    func getWorker(for index: Int) async -> ModelWorker {
        let workerID = index % 120
        if let existing = workerDict[workerID] { return existing }
        let newWorker = ModelWorker(id: workerID)
        workerDict[workerID] = newWorker
        return newWorker
    }
    func busyWorkerCount() async -> Int {
        var count = 0
        for worker in workerDict.values { if await worker.isBusy { count += 1 } }
        return count
    }
}

@available(macOS 26.0, *)
actor SchemaCache {
    private var cache: [String: String] = [:]
    func update(schema: String) { cache["current"] = schema }
    func get() -> String { return cache["current"] ?? "" }
}

@available(macOS 26.0, *)
final class ChatController: @unchecked Sendable {
    private let pool = ModelPool()
    private let schemaCache = SchemaCache()
    private var counter: Int = 0
    private let lock = NSLock()

    private func nextIndex() -> Int {
        lock.lock()
        defer { lock.unlock() }
        counter += 1
        return counter
    }
    
    @Sendable
    func createChatCompletion(req: Request) async throws -> Response {
        let chatReq = try req.content.decode(OpenAIChatRequest.self)
        var systemPart = ""
        var discoveryAnchor = ""
        var recentUserPart = ""
        var lastObservation = ""
        
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemPart = String((sys.content ?? "").prefix(800))
        }

        for msg in chatReq.messages {
            let content = msg.content ?? ""
            if content.contains("Field") && content.contains("Type") {
                let lines = content.components(separatedBy: "\n")
                var cols: [String] = []
                for line in lines {
                    let parts = line.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
                    if parts.count >= 2 && parts[0] != "Field" && !parts[0].contains("-") {
                        cols.append("\(parts[0]):\(parts[1])")
                    }
                }
                if !cols.isEmpty {
                    await schemaCache.update(schema: "SCHEMA: (\(cols.joined(separator: ", ")))")
                }
            }
        }
        
        discoveryAnchor = await schemaCache.get()

        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) {
            recentUserPart = String((lastUser.content ?? "").prefix(600))
        }
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) {
            lastObservation = String((lastObs.content ?? "").prefix(1000))
        }
        
        let prompt = """
        \(systemPart)
        STRUCTURE: \(discoveryAnchor)
        LAST: \(lastObservation)
        GOAL: \(recentUserPart)
        ACT NOW: [TOOL: name("args")]
        Assistant:
        """
        
        let workerIndex = nextIndex()
        let worker = await pool.getWorker(for: workerIndex)
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let busyCount = await pool.busyWorkerCount()
            let finalBusy = max(busyCount, 1)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            var toolCalls: [OpenAIToolCall] = []
            let pattern = "(?:\\[?TOOL:\\s*)?([a-zA-Z0-9_]+)\\s*\\((.*?)\\)\\]?"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let nsRange = NSRange(trimmed.startIndex..., in: trimmed)
                let matches = regex.matches(in: trimmed, options: [], range: nsRange)
                for match in matches {
                    if let nameRange = Range(match.range(at: 1), in: trimmed),
                       let argsRange = Range(match.range(at: 2), in: trimmed) {
                        let name = String(trimmed[nameRange]).trimmingCharacters(in: .whitespaces)
                        let argsRaw = String(trimmed[argsRange]).trimmingCharacters(in: .whitespaces)
                        var finalArgs = "{}"
                        if argsRaw.hasPrefix("{") { finalArgs = argsRaw } 
                        else {
                            let parts = argsRaw.components(separatedBy: ",").map { 
                                $0.trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
                            }
                            var dict: [String: String] = [:]
                            if name == "mariadb_query" { if parts.count >= 1 { dict["query"] = parts[0] } }
                            else if name == "mariadb_search" {
                                if parts.count >= 1 { dict["table"] = parts[0] }
                                if parts.count >= 2 { dict["query"] = parts[1] }
                            } else if name == "mariadb_sample" {
                                if parts.count >= 1 { dict["table"] = parts[0] }
                            } else if name == "think" {
                                if parts.count >= 1 { dict["note"] = parts[0] }
                            }
                            if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
                                finalArgs = s
                            }
                        }
                        toolCalls.append(OpenAIToolCall(id: "call_" + UUID().uuidString.prefix(8), type: "function", function: OpenAIToolCallFunction(name: name, arguments: finalArgs)))
                    }
                }
            }
            
            if toolCalls.isEmpty && (trimmed.contains("SELECT") || trimmed.contains("DESCRIBE")) {
                let sql = trimmed.components(separatedBy: "\n").first(where: { $0.contains("SELECT") || $0.contains("DESCRIBE") }) ?? trimmed
                let cleanSQL = sql.replacingOccurrences(of: "[TOOL: ", with: "").replacingOccurrences(of: "]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                let dict = ["query": cleanSQL]
                if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
                    toolCalls.append(OpenAIToolCall(id: "call_" + UUID().uuidString.prefix(8), type: "function", function: OpenAIToolCallFunction(name: "mariadb_query", arguments: s)))
                }
            }
            
            let responseObj = OpenAIChatResponse(
                id: "chatcmpl-" + UUID().uuidString,
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: chatReq.model,
                choices: [OpenAIChoice(index: 0, message: OpenAIMessageResponse(role: "assistant", content: toolCalls.isEmpty ? trimmed : nil, tool_calls: toolCalls.isEmpty ? nil : toolCalls), finish_reason: toolCalls.isEmpty ? "stop" : "tool_calls")],
                usage: OpenAIUsage(prompt_tokens: prompt.count/4, completion_tokens: trimmed.count/4, total_tokens: (prompt.count+trimmed.count)/4)
            )
            
            var rawRes = try JSONEncoder().encode(responseObj)
            if var json = try JSONSerialization.jsonObject(with: rawRes) as? [String: Any], var usage = json["usage"] as? [String: Any] {
                usage["active_workers"] = finalBusy
                usage["worker_id"] = workerIndex % 120
                json["usage"] = usage
                rawRes = try JSONSerialization.data(withJSONObject: json)
            }
            let res = Response(status: .ok, body: .init(data: rawRes))
            res.headers.replaceOrAdd(name: .contentType, value: "application/json")
            return res
        } catch {
            throw Abort(.internalServerError, reason: "Apple Bridge Error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String {
        struct SummarizeRequest: Content { let text: String }
        let summarizeReq = try req.content.decode(SummarizeRequest.self)
        let prompt = "Summarize: \(summarizeReq.text.prefix(2000))"
        let worker = await pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
