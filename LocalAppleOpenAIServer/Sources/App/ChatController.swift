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
        print("Worker [\(id)]: Prompt: \(prompt.count) chars")
        fflush(stdout)
        
        // Use a fresh session every time to prevent hidden context buildup.
        // This is crucial for OpenPlanter as it sends full history in its prompt.
        let session = LanguageModelSession(model: .default)
        let response = try await session.respond(to: prompt)
        
        print("Worker [\(id)]: Done.")
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
        let workerCount = min(Int(totalRAM / 2), cpuCores, 4)
        
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
        
        // --- DYNAMIC PROMPT RECONSTRUCTION (ULTRA-CONSERVATIVE) ---
        var systemPart = ""
        var toolsPart = ""
        var recentUserPart = ""
        var observationPart = ""
        
        // 1. System Prompt (strictly capped)
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemPart = String((sys.content ?? "").prefix(800))
        }
        
        // 2. Dynamic Tool Definitions (most important tools first)
        if let tools = chatReq.tools {
            toolsPart = "Available Tools:\n"
            for t in tools.prefix(8) { // Capped at 8 tools to save space
                let params = t.function.parameters?.stringify() ?? "{}"
                toolsPart += "- \(t.function.name)(\(params))\n"
            }
        }
        
        // 3. Most Recent User Instruction (the core command)
        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) {
            recentUserPart = String((lastUser.content ?? "").prefix(600))
        }
        
        // 4. Most Recent Observation (Tool Result) - HIGHEST PRIORITY DATA
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) {
            observationPart = String((lastObs.content ?? "").prefix(1200))
        }
        
        let prompt = """
        \(systemPart)
        
        \(toolsPart)
        
        STATE:
        \(observationPart)
        
        GOAL:
        \(recentUserPart)
        
        TASK: If STATE answers GOAL, say it. Else:
        CALL: tool_name({"arg": "val"})
        Assistant:
        """
        
        print("Final Reconstructed Prompt: \(prompt.count) chars")
        fflush(stdout)
        
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // --- DYNAMIC TOOL CALL PARSING ---
            var toolCalls: [OpenAIToolCall]? = nil
            
            // Regex to find "CALL: tool_name({json_args})"
            let pattern = "CALL:\\s*([a-zA-Z0-9_]+)\\s*\\((.*)\\)"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let range = NSRange(trimmed.startIndex..., in: trimmed)
                if let match = regex.firstMatch(in: trimmed, options: [], range: range) {
                    let nameRange = Range(match.range(at: 1), in: trimmed)!
                    let argsRange = Range(match.range(at: 2), in: trimmed)!
                    
                    let name = String(trimmed[nameRange])
                    let args = String(trimmed[argsRange])
                    
                    toolCalls = [OpenAIToolCall(
                        id: "call_" + UUID().uuidString.prefix(8),
                        type: "function",
                        function: OpenAIToolCallFunction(name: name, arguments: args)
                    )]
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
                        content: toolCalls == nil ? trimmed : nil,
                        tool_calls: toolCalls
                    ),
                    finish_reason: toolCalls == nil ? "stop" : "tool_calls"
                )],
                usage: OpenAIUsage(
                    prompt_tokens: prompt.count / 4,
                    completion_tokens: trimmed.count / 4,
                    total_tokens: (prompt.count + trimmed.count) / 4
                )
            )
            
            return try await responseObj.encodeResponse(for: req)
            
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
        let prompt = "Squeeze this data into one core fact sentence (max 150 chars): \(summarizeReq.text.prefix(2000))\nFact:"
        let worker = pool.getWorker(for: nextIndex())
        return try await worker.respond(to: prompt)
    }
}
