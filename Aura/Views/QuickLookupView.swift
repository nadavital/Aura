import SwiftUI

struct QuickLookupView: View {
    @EnvironmentObject var modelManager: MLXModelManager
    @StateObject private var llmService = LLMService()
    @State private var query: String = ""
    @State private var response: String = ""
    @State private var isLoading: Bool = false
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        VStack(spacing: 16) {
            // Search bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                
                TextField("Ask anything...", text: $query)
                    .textFieldStyle(.plain)
                    .font(.title2)
                    .onSubmit {
                        performSearch()
                    }
                
                if !query.isEmpty {
                    Button(action: { query = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding()
            .background(Color(.controlBackgroundColor))
            .cornerRadius(12)
            
            // Response area
            if isLoading {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Searching...")
                        .foregroundColor(.secondary)
                    Spacer()
                }
            } else if !response.isEmpty {
                ScrollView {
                    Text(response)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .background(Color(.textBackgroundColor))
                .cornerRadius(8)
            }
            
            Spacer()
            
            // Instructions
            if response.isEmpty && !isLoading {
                VStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .font(.largeTitle)
                        .foregroundColor(.secondary)
                    
                    Text("Quick AI Lookup")
                        .font(.headline)
                    
                    Text("Type your question and press Enter")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("Press Escape to close")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .frame(width: 600, height: 400)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .onAppear {
            llmService.setModelManager(modelManager)
            // Focus on the text field when the view appears
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                NSApp.keyWindow?.makeFirstResponder(nil)
            }
        }
    }
    
    private func performSearch() {
        guard !query.isEmpty else { return }
        
        isLoading = true
        response = ""
        
        Task {
            do {
                let result = try await llmService.sendQuickQuery(query)
                await MainActor.run {
                    response = result
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    response = "Error: \(error.localizedDescription)"
                    isLoading = false
                }
            }
        }
    }
}

extension NSApplication {
    static let keyboardShortcutNotification = Notification.Name("KeyboardShortcut")
}

#Preview {
    QuickLookupView()
}