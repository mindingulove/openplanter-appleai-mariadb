import Vapor

// MARK: - OpenAI Request Models
struct OpenAIChatRequest: Content {
    let model: String
    let messages: [OpenAIMessage]
    let temperature: Double?
    let stream: Bool?
}

struct OpenAIMessage: Content {
    let role: String
    let content: String? // Changed to optional to handle "nudges" or tool outputs
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
