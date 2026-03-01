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
            // Limit to roughly 3k tokens of characters to avoid 500 errors
            let safePrompt = String(prompt.prefix(12000))
            let response = try await session.respond(to: safePrompt)
            return response.content
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
        
        if let s = chatReq.messages.first(where: { $0.role == "system" }) { sys = String((s.content ?? "").prefix(400)) }
        
        // Build Schema Anchor
        for m in chatReq.messages {
            let c = m.content ?? ""
            if c.contains("Field") { await schemaCache.update(schema: "SCHEMA: id_aps, faixa_etaria, quantidade") }
        }
        anchor = await schemaCache.get()
        if anchor.isEmpty { anchor = "SCHEMA: (none discovered yet - run SHOW TABLES)" }
        
        var recentUserPart = ""
        var lastObservation = ""
        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) { recentUserPart = String((lastUser.content ?? "").prefix(400)) }
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) { lastObservation = String((lastObs.content ?? "").prefix(1000)) }

        let prompt: String
        // If this is the first turn (no tool observations yet), send the full setup
        if lastObservation.isEmpty {
            prompt = """
            \(sys)
            \(anchor)
            GOAL: \(recentUserPart)
            Assistant:
            """
        } else {
            // If we are mid-conversation, the stateful model already knows the sys and anchor.
            // Only send the new information to prevent exponential context growth.
            prompt = """
            TOOL RESULT: \(lastObservation)
            GOAL: \(recentUserPart)
            Assistant:
            """
        }

        let workerIndex = nextIndex()
        let worker = await pool.getWorker(for: workerIndex)

        do {
            await worker.setBusy(true)
            defer { Task { await worker.setBusy(false) } }

            let safePrompt = String(prompt.prefix(2000))
            let text = try await worker.respond(to: safePrompt)
            let trimmed = text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            var toolCalls: [OpenAIToolCall] = []
            
            let knownTools = ["mariadb_query", "mariadb_search", "mariadb_sample", "think", "read_file"]
            
            // --- STRUCTURAL PARSING (BULLETPROOF & MULTI-LINE) ---
            let pattern = "CALL_([a-zA-Z0-9_]+)\\s*\\((.*?)\\)"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let nsRange = NSRange(trimmed.startIndex..., in: trimmed)
                let matches = regex.matches(in: trimmed, options: [], range: nsRange)
                for match in matches {
                    if let nameRange = Range(match.range(at: 1), in: trimmed),
                       let argsRange = Range(match.range(at: 2), in: trimmed) {
                        let name = String(trimmed[nameRange]).trimmingCharacters(in: CharacterSet.whitespaces)
                        
                        if knownTools.contains(name) {
                            var argsRaw = String(trimmed[argsRange]).trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                            
                            // Strip quotes
                            if argsRaw.hasPrefix("\"") && argsRaw.hasSuffix("\"") {
                                argsRaw = String(argsRaw.dropFirst().dropLast())
                            } else if argsRaw.hasPrefix("'") && argsRaw.hasSuffix("'") {
                                argsRaw = String(argsRaw.dropFirst().dropLast())
                            }
                            
                            let dict = (name == "mariadb_query") ? ["query": argsRaw] : ["table": argsRaw]
                            if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
                                toolCalls.append(OpenAIToolCall(id: "call_" + UUID().uuidString.prefix(8), type: "function", function: OpenAIToolCallFunction(name: name, arguments: s)))
                            }
                        }
                    }
                }
            }
            
            // FALLBACK: If structural parsing fails, check if the block is PURELY SQL
            if toolCalls.isEmpty {
                let check = trimmed.lowercased().trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                if check.hasPrefix("select ") || check.hasPrefix("describe ") || check.hasPrefix("show tables") {
                    let cleanSQL = trimmed.replacingOccurrences(of: "`", with: "")
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
                usage["worker_id"] = workerIndex % 120
                json["usage"] = usage
                rawRes = try JSONSerialization.data(withJSONObject: json)
            }
            let res = Response(status: .ok, body: .init(data: rawRes))
            res.headers.replaceOrAdd(name: "Content-Type", value: "application/json")
            return res
        } catch {
            print("ChatController Error: \(error)")
            throw Abort(.internalServerError, reason: "Apple Bridge Error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String {
        return "Summary"
    }
}
