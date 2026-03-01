import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    let id: Int
    private var isBusy: Bool = false
    init(id: Int) { self.id = id }
    func setBusy(_ busy: Bool) { self.isBusy = busy }
    func getBusy() -> Bool { return self.isBusy }
    func respond(to prompt: String) async throws -> String {
        return try await Task.detached(priority: .userInitiated) {
            let session = LanguageModelSession(model: .default)
            return try await session.respond(to: String(prompt.prefix(10000))).content
        }.value
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
        for worker in workerDict.values { if await worker.getBusy() { count += 1 } }
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
        lock.lock(); defer { lock.unlock() }; counter += 1; return counter
    }
    
    @Sendable
    func createChatCompletion(req: Request) async throws -> Response {
        let chatReq = try req.content.decode(OpenAIChatRequest.self)
        var sys = "", anchor = "", goal = "", last = ""
        
        if let s = chatReq.messages.first(where: { $0.role == "system" }) { 
            sys = String((s.content ?? "").prefix(800)) 
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
                    await schemaCache.update(schema: "STRUCTURE: ( \(cols.joined(separator: ", ")) )")
                }
            }
        }
        
        anchor = await schemaCache.get()
        if anchor.isEmpty { anchor = "STRUCTURE: (id_aps, faixa_etaria, quantidade)" }

        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) { 
            goal = String((lastUser.content ?? "").prefix(600)) 
        }
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) { 
            last = String((lastObs.content ?? "").prefix(1000)) 
        }
        
        let prompt = """
        \(sys)
        \(anchor)
        LATEST: \(last)
        GOAL: \(goal)
        ACT: Output [TOOL: mariadb_query("SQL")]
        Assistant:
        """
        
        let workerIndex = nextIndex()
        let workerID = workerIndex % 120
        let worker = await pool.getWorker(for: workerIndex)
        
        do {
            await worker.setBusy(true)
            defer { Task { await worker.setBusy(false) } }
            
            let text = try await worker.respond(to: prompt)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            
            var toolCalls: [OpenAIToolCall] = []
            let knownTools = ["mariadb_query", "mariadb_search", "mariadb_sample", "think", "read_file"]
            
            let pattern = "(?:\\[?TOOL:\\s*)?(?:name|tool)?[:\\s]*([a-zA-Z0-9_]+)\\s*\\((.*?)\\)\\]?"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let nsRange = NSRange(trimmed.startIndex..., in: trimmed)
                let matches = regex.matches(in: trimmed, options: [], range: nsRange)
                for match in matches {
                    if let nameRange = Range(match.range(at: 1), in: trimmed),
                       let argsRange = Range(match.range(at: 2), in: trimmed) {
                        
                        let name = String(trimmed[nameRange]).trimmingCharacters(in: .whitespaces)
                        if !knownTools.contains(name) { continue }
                        
                        let argsRaw = String(trimmed[argsRange]).trimmingCharacters(in: .whitespaces)
                        var finalArgs = "{}"
                        
                        if argsRaw.hasPrefix("{") { finalArgs = argsRaw } 
                        else {
                            let parts = argsRaw.components(separatedBy: ",").map { 
                                $0.trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
                            }
                            var dict: [String: String] = [:]
                            if name == "mariadb_query" { 
                                var sql = parts[0]
                                if !sql.lowercased().contains("from") { sql += " FROM vw_aps_faixa_etaria" }
                                dict["query"] = sql
                            }
                            else if name == "mariadb_sample" { dict["table"] = parts[0] }
                            else if name == "think" { dict["note"] = parts[0] }
                            
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
                if !cleanSQL.isEmpty {
                    let dict = ["query": cleanSQL]
                    if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
                        toolCalls.append(OpenAIToolCall(id: "call_" + UUID().uuidString.prefix(8), type: "function", function: OpenAIToolCallFunction(name: "mariadb_query", arguments: s)))
                    }
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
            
            let busyCount = await pool.busyWorkerCount()
            var rawRes = try JSONEncoder().encode(responseObj)
            if var json = try JSONSerialization.jsonObject(with: rawRes) as? [String: Any], var usage = json["usage"] as? [String: Any] {
                usage["active_workers"] = max(busyCount, 1)
                usage["worker_id"] = workerID
                json["usage"] = usage
                rawRes = try JSONSerialization.data(withJSONObject: json)
            }
            let res = Response(status: .ok, body: .init(data: rawRes))
            res.headers.replaceOrAdd(name: .contentType, value: "application/json")
            return res
        } catch {
            print("ChatController Error: \(error)")
            throw Abort(.internalServerError, reason: "Apple Bridge Error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String { return "Summary" }
}
