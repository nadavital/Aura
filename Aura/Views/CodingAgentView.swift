import SwiftUI

struct CodingAgentView: View {
    @EnvironmentObject var modelManager: MLXModelManager
    @StateObject private var llmService = LLMService()
    @StateObject private var toolExecutor = ToolExecutor()
    @State private var inputText: String = ""
    @State private var output: String = ""
    @State private var isLoading: Bool = false
    @State private var workingDirectory: String = NSHomeDirectory()
    
    var body: some View {
        VStack(spacing: 0) {
            // Working directory bar
            HStack {
                Text("Working Directory:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(workingDirectory)
                    .font(.caption.monospaced())
                    .foregroundColor(.primary)
                    .lineLimit(1)
                
                Spacer()
                
                Button("Change") {
                    selectWorkingDirectory()
                }
                .font(.caption)
            }
            .padding()
            .background(Color(.controlBackgroundColor))
            
            Divider()
            
            // Output area
            ScrollView {
                Text(output.isEmpty ? "Ask me to help with coding tasks...\n\nExamples:\n• \"Create a Python script to analyze CSV files\"\n• \"Debug this JavaScript function\"\n• \"Refactor this code for better performance\"" : output)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .textSelection(.enabled)
            }
            .background(Color(.textBackgroundColor))
            
            Divider()
            
            // Input area
            VStack {
                HStack {
                    TextField("Describe your coding task...", text: $inputText, axis: .vertical)
                        .textFieldStyle(.plain)
                        .lineLimit(1...5)
                        .onSubmit {
                            processTask()
                        }
                    
                    Button(action: processTask) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundColor(inputText.isEmpty ? .gray : .accentColor)
                    }
                    .disabled(inputText.isEmpty || isLoading)
                }
                
                if isLoading {
                    HStack {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Processing...")
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                    .padding(.top, 4)
                }
            }
            .padding()
        }
        .navigationTitle("Coding Agent")
        .onAppear {
            llmService.setModelManager(modelManager)
        }
    }
    
    private func processTask() {
        let task = inputText
        inputText = ""
        isLoading = true
        
        output += "\n🤖 Processing: \(task)\n\n"
        
        Task {
            do {
                let systemPrompt = """
                You are a coding assistant with access to file system tools. Help the user with their coding task. 
                Working directory: \(workingDirectory)
                
                Available tools:
                - Execute shell commands
                - Read/write files  
                - List directories
                - Create directories
                
                Provide clear, executable code and explanations. When possible, actually execute the tools needed.
                """
                
                let messages = [
                    Message(content: systemPrompt, isUser: false),
                    Message(content: task, isUser: true)
                ]
                
                let response = try await llmService.sendMessage(messages)
                await MainActor.run {
                    output += response
                }
                
                await MainActor.run {
                    output += "\n\n"
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    output += "❌ Error: \(error.localizedDescription)\n\n"
                    isLoading = false
                }
            }
        }
    }
    
    private func selectWorkingDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK {
            if let url = panel.url {
                workingDirectory = url.path
            }
        }
    }
}

#Preview {
    CodingAgentView()
}