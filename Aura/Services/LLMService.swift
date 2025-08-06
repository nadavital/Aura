import Foundation

@MainActor
class LLMService: ObservableObject {
    private weak var modelManager: MLXModelManager?
    
    init(modelManager: MLXModelManager? = nil) {
        self.modelManager = modelManager
    }
    
    func setModelManager(_ modelManager: MLXModelManager) {
        self.modelManager = modelManager
    }
    
    func sendMessage(_ messages: [Message], model: String = "gpt-oss-20b") async throws -> String {
        guard let modelManager = modelManager else {
            throw LLMError.modelManagerNotSet
        }
        
        return try await modelManager.generateResponse(for: messages)
    }
    
    func sendQuickQuery(_ query: String) async throws -> String {
        guard let modelManager = modelManager else {
            throw LLMError.modelManagerNotSet
        }
        
        return try await modelManager.generateQuickResponse(for: query)
    }
}

enum LLMError: Error {
    case invalidURL
    case noResponse
    case decodingError
    case modelManagerNotSet
}
