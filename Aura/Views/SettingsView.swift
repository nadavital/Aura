import SwiftUI

struct SettingsView: View {
    @StateObject private var mlxModelManager = MLXModelManager()
    @AppStorage("quickLookupEnabled") private var quickLookupEnabled: Bool = true
    @AppStorage("globalShortcutEnabled") private var globalShortcutEnabled: Bool = true
    
    @State private var showingAlert = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    
    var body: some View {
        Form {
            Section("MLX Local AI Model") {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Model:")
                        Spacer()
                        Text("GPT-OSS-20B-MLX-8bit")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Status:")
                        Spacer()
                        modelStatusView
                    }
                    
                    if mlxModelManager.modelState == .downloading {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Download Progress:")
                                Spacer()
                                Text("\(Int(mlxModelManager.downloadProgress * 100))%")
                                    .foregroundColor(.secondary)
                            }
                            
                            ProgressView(value: mlxModelManager.downloadProgress)
                                .progressViewStyle(LinearProgressViewStyle())
                            
                            HStack {
                                Text("Downloaded:")
                                Spacer()
                                Text(String(format: "%.1f GB", mlxModelManager.downloadedGB))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    if let error = mlxModelManager.downloadError {
                        Text("Error: \(error)")
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                }
                
                VStack(spacing: 8) {
                    modelActionButtons
                }
            }
            
            Section("Quick Lookup") {
                Toggle("Enable Quick Lookup", isOn: $quickLookupEnabled)
                Toggle("Global Shortcut (⌘⇧Space)", isOn: $globalShortcutEnabled)
                    .disabled(!quickLookupEnabled)
                
                Text("Use ⌘⇧Space to open quick lookup from anywhere")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Section("About") {
                HStack {
                    Text("Version:")
                    Spacer()
                    Text("1.0.0")
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text("Build:")
                    Spacer()
                    Text("1")
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text("AI Engine:")
                    Spacer()
                    Text("MLX Apple Silicon")
                        .foregroundColor(.secondary)
                }
            }
        }
        .navigationTitle("Settings")
        .formStyle(.grouped)
        .alert(alertTitle, isPresented: $showingAlert) {
            Button("OK") { }
        } message: {
            Text(alertMessage)
        }
    }
    
    private var modelStatusView: some View {
        Group {
            switch mlxModelManager.modelState {
            case .checking:
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Checking...")
                }
                .foregroundColor(.secondary)
            case .notDownloaded:
                Text("Not Downloaded")
                    .foregroundColor(.orange)
            case .downloading:
                Text("Downloading...")
                    .foregroundColor(.blue)
            case .ready:
                Text("Ready")
                    .foregroundColor(.green)
            case .corrupted:
                Text("Corrupted")
                    .foregroundColor(.red)
            case .downloadFailed:
                Text("Download Failed")
                    .foregroundColor(.red)
            }
        }
    }
    
    private var modelActionButtons: some View {
        VStack(spacing: 8) {
            switch mlxModelManager.modelState {
            case .notDownloaded, .downloadFailed:
                Button("Download Model (22 GB)") {
                    downloadModel()
                }
                .buttonStyle(.borderedProminent)
                .disabled(mlxModelManager.isDownloading)
                
            case .ready:
                HStack(spacing: 12) {
                    Button("Test Model") {
                        testModel()
                    }
                    .buttonStyle(.bordered)
                    
                    Button("Re-download") {
                        redownloadModel()
                    }
                    .buttonStyle(.bordered)
                    
                    Button("Delete Model") {
                        deleteModel()
                    }
                    .buttonStyle(.bordered)
                    .foregroundColor(.red)
                }
                
            case .corrupted:
                VStack(spacing: 8) {
                    Button("Re-download Model") {
                        redownloadModel()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(mlxModelManager.isDownloading)
                    
                    Button("Delete Corrupted Model") {
                        deleteModel()
                    }
                    .buttonStyle(.bordered)
                    .foregroundColor(.red)
                }
                
            case .downloading:
                Button("Cancel Download") {
                    // TODO: Implement cancel functionality
                }
                .buttonStyle(.bordered)
                .foregroundColor(.red)
                .disabled(true) // Disable for now
                
            case .checking:
                EmptyView()
            }
        }
    }
    
    private func downloadModel() {
        Task {
            do {
                try await mlxModelManager.downloadModel()
                await MainActor.run {
                    showAlert(title: "Success", message: "Model downloaded successfully!")
                }
            } catch {
                await MainActor.run {
                    showAlert(title: "Download Failed", message: error.localizedDescription)
                }
            }
        }
    }
    
    private func testModel() {
        Task {
            do {
                let response = try await mlxModelManager.generateQuickResponse(for: "Hello! Please respond with a brief test message.")
                await MainActor.run {
                    showAlert(title: "Model Test", message: "Model is working correctly!\n\nResponse: \(response.prefix(100))...")
                }
            } catch {
                await MainActor.run {
                    showAlert(title: "Test Failed", message: error.localizedDescription)
                }
            }
        }
    }
    
    private func redownloadModel() {
        Task {
            do {
                try await mlxModelManager.deleteAndRedownloadModel()
                await MainActor.run {
                    showAlert(title: "Success", message: "Model re-downloaded successfully!")
                }
            } catch {
                await MainActor.run {
                    showAlert(title: "Re-download Failed", message: error.localizedDescription)
                }
            }
        }
    }
    
    private func deleteModel() {
        Task {
            do {
                try mlxModelManager.deleteModel()
                await MainActor.run {
                    showAlert(title: "Success", message: "Model deleted successfully!")
                }
            } catch {
                await MainActor.run {
                    showAlert(title: "Delete Failed", message: error.localizedDescription)
                }
            }
        }
    }
    
    private func showAlert(title: String, message: String) {
        alertTitle = title
        alertMessage = message
        showingAlert = true
    }
}

#Preview {
    SettingsView()
}
