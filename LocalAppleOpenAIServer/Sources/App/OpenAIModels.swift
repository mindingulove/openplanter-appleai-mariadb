import Vapor

// MARK: - OpenAI Request Models
struct OpenAIChatRequest: Content {
    let model: String
    let messages: [OpenAIMessage]
    let temperature: Double?
    let stream: Bool?
    let tools: [OpenAITool]?
}

struct OpenAITool: Content {
    let type: String
    let function: OpenAIFunction
}

struct OpenAIFunction: Content {
    let name: String
    let description: String?
    let parameters: [String: AnyJSON]?
}

struct AnyJSON: Content {
    let value: Any
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(String.self) { value = v }
        else if let v = try? container.decode(Int.self) { value = v }
        else if let v = try? container.decode(Double.self) { value = v }
        else if let v = try? container.decode(Bool.self) { value = v }
        else if let v = try? container.decode([String: AnyJSON].self) { value = v }
        else if let v = try? container.decode([AnyJSON].self) { value = v }
        else { value = NSNull() }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let v = value as? String { try container.encode(v) }
        else if let v = value as? Int { try container.encode(v) }
        else if let v = value as? Double { try container.encode(v) }
        else if let v = value as? Bool { try container.encode(v) }
        else if let v = value as? [String: AnyJSON] { try container.encode(v) }
        else if let v = value as? [AnyJSON] { try container.encode(v) }
    }
}

struct OpenAIMessage: Content {
    let role: String
    let content: String?
    let tool_calls: [OpenAIToolCall]?
}

struct OpenAIToolCall: Content {
    let id: String
    let type: String
    let function: OpenAIToolCallFunction
}

struct OpenAIToolCallFunction: Content {
    let name: String
    let arguments: String
}

// MARK: - OpenAI Response Models
struct OpenAIChatResponse: Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [OpenAIChoice]
    let usage: OpenAIUsage
}

struct OpenAIChoice: Content {
    let index: Int
    let message: OpenAIMessage
    let finish_reason: String?
}

struct OpenAIUsage: Content {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}
