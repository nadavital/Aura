import Foundation

struct Message: Identifiable, Codable {
    let id = UUID()
    let content: String
    let isUser: Bool
    let timestamp: Date
    
    init(content: String, isUser: Bool) {
        self.content = content
        self.isUser = isUser
        self.timestamp = Date()
    }
}

struct ChatRequest: Codable {
    let messages: [ChatMessage]
    let model: String
    let temperature: Double
    let stream: Bool
    
    init(messages: [Message], model: String = "gpt-4", temperature: Double = 0.7, stream: Bool = true) {
        self.messages = messages.map { message in
            ChatMessage(
                role: message.isUser ? "user" : "assistant",
                content: message.content
            )
        }
        self.model = model
        self.temperature = temperature
        self.stream = stream
    }
}

struct ChatMessage: Codable {
    let role: String
    let content: String
}

struct ChatResponse: Codable {
    let choices: [ChatChoice]
}

struct ChatChoice: Codable {
    let delta: ChatDelta?
    let message: ChatMessage?
}

struct ChatDelta: Codable {
    let content: String?
}