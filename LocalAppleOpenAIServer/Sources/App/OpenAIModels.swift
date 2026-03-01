import Vapor

// MARK: - JSON Value for Sendable/Codable
enum JSONValue: Codable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(String.self) { self = .string(v) }
        else if let v = try? container.decode(Double.self) { self = .number(v) }
        else if let v = try? container.decode(Bool.self) { self = .bool(v) }
        else if let v = try? container.decode([JSONValue].self) { self = .array(v) }
        else if let v = try? container.decode([String: JSONValue].self) { self = .object(v) }
        else { self = .null }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .number(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        case .array(let v): try container.encode(v)
        case .object(let v): try container.encode(v)
        case .null: try container.encodeNil()
        }
    }
    
    func stringify() -> String {
        switch self {
        case .string(let v): return "\"\(v)\""
        case .number(let v): return "\(v)"
        case .bool(let v): return "\(v)"
        case .array(let v): return "[" + v.map { $0.stringify() }.joined(separator: ", ") + "]"
        case .object(let v): return "{" + v.map { "\"\($0.key)\": \($0.value.stringify())" }.joined(separator: ", ") + "}"
        case .null: return "null"
        }
    }
}

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
    let parameters: JSONValue?
}

struct OpenAIMessage: Content {
    let role: String
    let content: String?
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
    let message: OpenAIMessageResponse
    let finish_reason: String?
}

struct OpenAIMessageResponse: Content {
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

struct OpenAIUsage: Content {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}
