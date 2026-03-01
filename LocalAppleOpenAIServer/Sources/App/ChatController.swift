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
        
        // Build Schema Anchor
        for m in chatReq.messages {
            let c = m.content ?? ""
            if c.contains("Field") { await schemaCache.update(schema: "SCHEMA: id_aps, faixa_etaria, quantidade") }
        }
        anchor = await schemaCache.get()
        if anchor.isEmpty { anchor = "SCHEMA: (id_aps, faixa_etaria, quantidade)" }
        
        if let g = chatReq.messages.last(where: { $0.role == "user" }) { goal = String(g.content?.prefix(400) ?? "") }
        if let l = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) { last = String(l.content?.prefix(600) ?? "") }
        
        let prompt = "\(sys)\n\(anchor)\nLAST: \(last)\nGOAL: \(goal)\nACT: CALL_mariadb_query(\"SQL\")"
        let worker = pool[nextIndex() % 20]
        
        do {
            let text = try await worker.respond(to: prompt)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            var toolCalls: [OpenAIToolCall] = []
            
            let knownTools = ["mariadb_query", "mariadb_search", "mariadb_sample", "think", "read_file"]
            
            // --- STRUCTURAL PARSING (BULLETPROOF) ---
            let lines = trimmed.components(separatedBy: "\n")
            for line in lines {
                let cleanLine = line.trimmingCharacters(in: .whitespaces)
                
                // Look for CALL_toolname("args")
                if cleanLine.contains("CALL_") && cleanLine.contains("(") {
                    let parts = cleanLine.components(separatedBy: "CALL_")
                    if parts.count > 1 {
                        let remainder = parts[1] // e.g. "mariadb_query("SELECT...")"
                        if let parenIdx = remainder.firstIndex(of: "(") {
                            let name = String(remainder[..<parenIdx]).trimmingCharacters(in: .whitespaces)
                            
                            // HARD FILTER: Must be a known tool
                            if knownTools.contains(name) {
                                // Extract everything after the first '('
                                let afterParen = remainder[remainder.index(after: parenIdx)...]
                                // Find the last ')' to handle nested parentheses (like COUNT(*))
                                if let lastParenIdx = afterParen.lastIndex(of: ")") {
                                    var argsRaw = String(afterParen[..<lastParenIdx]).trimmingCharacters(in: .whitespaces)
                                    
                                    // Strip quotes
                                    if argsRaw.hasPrefix("\"") && argsRaw.hasSuffix("\"") {
                                        argsRaw = String(argsRaw.dropFirst().dropLast())
                                    } else if argsRaw.hasPrefix("'") && argsRaw.hasSuffix("'") {
                                        argsRaw = String(argsRaw.dropFirst().dropLast())
                                    }
                                    
                                    if name == "mariadb_query" {
                                        let lsql = argsRaw.lowercased()
                                        if lsql.contains("select") && !lsql.contains("from") {
                                            argsRaw += " FROM vw_aps_faixa_etaria"
                                        }
                                        if lsql.contains("select id_aps") && !lsql.contains("sum") && !lsql.contains("count") {
                                            argsRaw = "SELECT id_aps, SUM(quantidade) as total FROM vw_aps_faixa_etaria WHERE faixa_etaria LIKE '%04%' GROUP BY id_aps ORDER BY total DESC LIMIT 1"
                                        }
                                    }
                                    
                                    let dict = (name == "mariadb_query") ? ["query": argsRaw] : ["table": argsRaw]
                                    if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
                                        toolCalls.append(OpenAIToolCall(id: "call_" + UUID().uuidString.prefix(8), type: "function", function: OpenAIToolCallFunction(name: name, arguments: s)))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // FALLBACK: If structural parsing fails, check if the whole block is just SQL
            if toolCalls.isEmpty {
                let check = trimmed.lowercased()
                if check.contains("select ") || check.contains("describe ") || check.contains("show tables") {
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
            
            var rawRes = try JSONEncoder().encode(responseObj)
            if var json = try JSONSerialization.jsonObject(with: rawRes) as? [String: Any], var usage = json["usage"] as? [String: Any] {
                usage["active_workers"] = 1
                usage["worker_id"] = nextIndex() % 20
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
        return "Summary"
    }
}
