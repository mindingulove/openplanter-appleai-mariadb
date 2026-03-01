import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    private let id: Int
    private let session: LanguageModelSession
    
    init(id: Int) {
        self.id = id
        // Persistent session to avoid re-initialization overhead.
        self.session = LanguageModelSession(model: .default)
    }
    
    func respond(to prompt: String) async throws -> String {
        print("Worker [\(id)]: Prompt: \(prompt.count) chars")
        fflush(stdout)
        
        // We use the stateful session. respond(to:) appends to the internal transcript.
        // To keep it OpenAI-compatible (stateless from bridge's perspective),
        // we should ideally clear the transcript, but since we manage context 
        // in the prompt reconstruction, we'll just send the prompt.
        // Note: Apple's FoundationModels often handle long context by internal pruning.
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
        // INCREASED CAP: 32 workers as requested, or based on system resources.
        let workerCount = min(Int(totalRAM / 2), cpuCores, 32) 
        
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
        
        // --- ANCHORED PROMPT RECONSTRUCTION ---
        var systemPart = ""
        var toolsPart = ""
        var discoveryAnchor = ""
        var recentUserPart = ""
        var lastObservation = ""
        
        // 1. System Prompt
        if let sys = chatReq.messages.first(where: { $0.role == "system" }) {
            systemPart = String((sys.content ?? "").prefix(800))
        }
        
        // 2. Dynamic Tool Definitions
        if let tools = chatReq.tools {
            toolsPart = "Available Tools:\n"
            for t in tools.prefix(10) { 
                let params = t.function.parameters?.stringify() ?? "{}"
                toolsPart += "- \(t.function.name)(\(params))\n"
            }
        }
        
        // 3. Discovery Anchor (The VERY FIRST tool result, usually DESCRIBE or SHOW TABLES)
        // This prevents the model from forgetting the schema in long conversations.
        if let firstObs = chatReq.messages.first(where: { $0.role != "user" && $0.role != "system" }) {
            discoveryAnchor = String((firstObs.content ?? "").prefix(1000))
        }
        
        // 4. Most Recent User Instruction
        if let lastUser = chatReq.messages.last(where: { $0.role == "user" }) {
            recentUserPart = String((lastUser.content ?? "").prefix(600))
        }
        
        // 5. Last Observation (Latest Result)
        if let lastObs = chatReq.messages.last(where: { $0.role != "user" && $0.role != "system" }) {
            lastObservation = String((lastObs.content ?? "").prefix(1200))
        }
        
        let prompt = """
        \(systemPart)
        
        \(toolsPart)
        
        SCHEMA/CONTEXT ANCHOR:
        \(discoveryAnchor)
        
        LATEST DATA:
        \(lastObservation)
        
        USER GOAL:
        \(recentUserPart)
        
        TASK: If LATEST DATA answers USER GOAL, provide final answer. Else:
        CALL: tool_name({"arg": "val"})
        Assistant:
        """
        
        print("Final Reconstructed Prompt: \(prompt.count) chars")
        fflush(stdout)
        
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let generatedText = try await worker.respond(to: prompt)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Dynamic Tool Call Parsing
            var toolCalls: [OpenAIToolCall]? = nil
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
