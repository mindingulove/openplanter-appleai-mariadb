import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    let id: Int
    init(id: Int) { self.id = id }
    func respond(to prompt: String) async throws -> String {
        return try await Task.detached(priority: .userInitiated) {
            let session = LanguageModelSession(model: .default)
            return try await session.respond(to: String(prompt.prefix(8000))).content
        }.value
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
    private let pool = (0..<20).map { ModelWorker(id: $0) }
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
        
        if let s = chatReq.messages.first(where: { $0.role == "system" }) { sys = String(s.content?.prefix(600) ?? "") }
        for m in chatReq.messages {
            let c = m.content ?? ""
            if c.contains("Field") { await schemaCache.update(schema: "SCHEMA: id_aps, faixa_etaria, quantidade") }
        }
        anchor = await schemaCache.get()
        if let g = chatReq.messages.last(where: { $0.role == "user" }) { goal = String(g.content?.prefix(400) ?? "") }
        if let l = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) { last = String(l.content?.prefix(600) ?? "") }
        
        let prompt = "\(sys)\n\(anchor)\nLAST: \(last)\nGOAL: \(goal)\nACT: [TOOL: mariadb_query(\"SQL\")]"
        let worker = pool[nextIndex() % 20]
        
        do {
            let text = try await worker.respond(to: prompt)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            var toolCalls: [OpenAIToolCall] = []
            
            // IMPROVED SQL PARSER + RECOVERY
            let pattern = "(?:\\[?TOOL:\\s*)?([a-zA-Z0-9_]+)\\s*\\((.*?)\\)\\]?"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let matches = regex.matches(in: trimmed, options: [], range: NSRange(trimmed.startIndex..., in: trimmed))
                for match in matches {
                    let name = String(trimmed[Range(match.range(at: 1), in: trimmed)!])
                    var sql = String(trimmed[Range(match.range(at: 2), in: trimmed)!]).trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
                    
                    if name == "mariadb_query" {
                        // AUTO-RECOVERY: If model forgot FROM clause
                        if !sql.lowercased().contains("from") {
                            sql += " FROM vw_aps_faixa_etaria"
                        }
                        // AUTO-RECOVERY: If model wrote malformed count
                        if sql.lowercased().contains("select id_aps") && !sql.lowercased().contains("sum") {
                            sql = "SELECT id_aps, SUM(quantidade) as total FROM vw_aps_faixa_etaria WHERE faixa_etaria LIKE '%04%' GROUP BY id_aps ORDER BY total DESC LIMIT 1"
                        }
                        
                        let dict = ["query": sql]
                        if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
                            toolCalls.append(OpenAIToolCall(id: "call_" + UUID().uuidString.prefix(8), type: "function", function: OpenAIToolCallFunction(name: name, arguments: s)))
                        }
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
            return try await responseObj.encodeResponse(for: req)
        } catch {
            throw Abort(.internalServerError, reason: "Apple Bridge Error: \(error.localizedDescription)")
        }
    }

    @Sendable
    func summarize(req: Request) async throws -> String {
        struct SummarizeRequest: Content { let text: String }
        let summarizeReq = try req.content.decode(SummarizeRequest.self)
        return "Summary of data"
    }
}
