import Vapor
import Foundation
import FoundationModels

@available(macOS 26.0, *)
actor ModelWorker {
    private let id: Int
    private let session: LanguageModelSession
    
    init(id: Int) {
        self.id = id
        self.session = LanguageModelSession(model: .default)
    }
    
    func respond(to messages: [LanguageModelSession.Message], tools: [ChatTool]? = nil) async throws -> LanguageModelSession.Response {
        print("Worker [\(id)]: Messages: \(messages.count), Tools: \(tools?.count ?? 0)")
        fflush(stdout)
        
        // Pass tools natively to the Foundation Model session
        let response = try await session.respond(to: messages, tools: tools ?? [])
        
        print("Worker [\(id)]: Done. Content: \(response.content.count) chars")
        fflush(stdout)
        return response
    }
}

@available(macOS 26.0, *)
final class ModelPool: Sendable {
    private let workers: [ModelWorker]
    
    init() {
        let totalRAM = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let cpuCores = ProcessInfo.processInfo.processorCount
        let workerCount = min(Int(totalRAM / 2), cpuCores, 4) // Capped at 4 for stability
        
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
        
        // Translate OpenAI messages to Apple Foundation Model messages
        let messages: [LanguageModelSession.Message] = chatReq.messages.compactMap { msg in
            let role: LanguageModelSession.Message.Role
            switch msg.role {
            case "system": role = .system
            case "user": role = .user
            case "assistant": role = .assistant
            case "tool": role = .tool
            default: return nil
            }
            return LanguageModelSession.Message(role: role, content: msg.content ?? "")
        }
        
        // Translate OpenAI tools to Apple Foundation Model tools
        let chatTools: [ChatTool]? = chatReq.tools?.compactMap { tool in
            guard tool.type == "function" else { return nil }
            return ChatTool(
                name: tool.function.name,
                description: tool.function.description ?? "",
                parameters: tool.function.parameters // Simplified mapping
            )
        }
        
        let worker = pool.getWorker(for: nextIndex())
        
        do {
            let modelResponse = try await worker.respond(to: messages, tools: chatTools)
            
            // Map native tool calls back to OpenAI format
            var openAIToolCalls: [OpenAIToolCall]? = nil
            if let nativeCalls = modelResponse.toolCalls {
                openAIToolCalls = nativeCalls.map { call in
                    OpenAIToolCall(
                        id: "call_" + UUID().uuidString.prefix(8),
                        type: "function",
                        function: OpenAIToolCallFunction(
                            name: call.name,
                            arguments: call.argumentsJSON
                        )
                    )
                }
            }
            
            let responseObj: [String: Any] = [
                "id": "chatcmpl-\(UUID().uuidString)",
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": chatReq.model,
                "choices": [[
                    "index": 0,
                    "message": [
                        "role": "assistant",
                        "content": modelResponse.content,
                        "tool_calls": openAIToolCalls != nil ? try JSONSerialization.jsonObject(with: JSONEncoder().encode(openAIToolCalls)) : nil
                    ],
                    "finish_reason": openAIToolCalls != nil ? "tool_calls" : "stop"
                ]],
                "usage": [
                    "prompt_tokens": 0, // Simplified for now
                    "completion_tokens": modelResponse.content.count / 4,
                    "total_tokens": modelResponse.content.count / 4
                ]
            ]
            
            let data = try JSONSerialization.data(withJSONObject: responseObj)
            let res = Response(status: .ok, body: .init(data: data))
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
        let messages = [LanguageModelSession.Message(role: .user, content: "Squeeze this data into one core fact sentence (max 150 chars): \(summarizeReq.text.prefix(2000))\nFact:")]
        let worker = pool.getWorker(for: nextIndex())
        let response = try await worker.respond(to: messages)
        return response.content
    }
}

