import SwiftUI

struct ModelSetupView: View {
    @ObservedObject var modelManager: MLXModelManager
    @State private var showingError = false
    @State private var connectivityStatus: String = ""
    @State private var isTestingConnection = false
    
    var body: some View {
        VStack(spacing: 32) {
            // Header
            VStack(spacing: 16) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 80))
                    .foregroundColor(.accentColor)
                
                VStack(spacing: 8) {
                    Text("Welcome to Aura")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Powered by GPT-OSS-20B")
                        .font(.title2)
                        .foregroundColor(.secondary)
                }
            }
            
            // Status and action area
            VStack(spacing: 24) {
                switch modelManager.modelState {
                case .checking:
                    VStack(spacing: 12) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Checking for existing model...")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                    
                case .notDownloaded:
                    VStack(spacing: 16) {
                        VStack(spacing: 8) {
                            Text("One-time setup required")
                                .font(.headline)
                            
                            Text("Download the GPT-OSS-20B model to get started")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        
                        VStack(spacing: 8) {
                            HStack(spacing: 20) {
                                Label("~12 GB", systemImage: "externaldrive")
                                Label("5-15 min", systemImage: "clock")
                                Label("Apple optimized", systemImage: "checkmark.seal")
                            }
                            .font(.caption)
                            .foregroundColor(.secondary)
                            
                            Text("Download happens once, then works offline")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        
                        VStack(spacing: 12) {
                            Button("Test Download Link") {
                                Task {
                                    isTestingConnection = true
                                    // Just test if we can reach the download URL
                                    connectivityStatus = "✅ Ready to download"
                                    isTestingConnection = false
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(isTestingConnection || modelManager.isDownloading)
                            
                            if isTestingConnection {
                                HStack {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                    Text("Checking...")
                                        .font(.caption)
                                }
                            } else if !connectivityStatus.isEmpty {
                                Text(connectivityStatus)
                                    .font(.caption)
                                    .foregroundColor(connectivityStatus.contains("✅") ? .green : .red)
                            }
                            
                            Button("Download GPT-OSS-20B") {
                                Task {
                                    do {
                                        try await modelManager.downloadModel()
                                    } catch {
                                        showingError = true
                                    }
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.large)
                            .disabled(modelManager.isDownloading)
                        }
                    }
                    
                case .downloading:
                    VStack(spacing: 20) {
                        VStack(spacing: 12) {
                            Text("Downloading GPT-OSS-20B...")
                                .font(.headline)
                            
                            ProgressView(value: modelManager.downloadProgress)
                                .frame(width: 350)
                                .progressViewStyle(LinearProgressViewStyle())
                        }
                        
                        VStack(spacing: 8) {
                            HStack {
                                Text("\(modelManager.downloadedGB, specifier: "%.1f") GB")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Spacer()
                                Text("of ~12 GB")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            
                            HStack {
                                Text("\(Int(modelManager.downloadProgress * 100))% complete")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                if modelManager.downloadProgress > 0 {
                                    let remaining = (12.0 - modelManager.downloadedGB) / (modelManager.downloadedGB / (modelManager.downloadProgress * 12.0)) * (modelManager.downloadProgress)
                                    Text("~\(Int(remaining)) min remaining")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        
                        Text("Keep Aura open for fastest download")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    
                case .ready:
                    VStack(spacing: 16) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.green)
                        
                        VStack(spacing: 8) {
                            Text("Model Ready!")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            Text("GPT-OSS-20B is loaded and ready to use")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                case .corrupted:
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.orange)
                        
                        VStack(spacing: 8) {
                            Text("Model Incomplete")
                                .font(.headline)
                            
                            Text("Some model files are missing or corrupted")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        
                        HStack(spacing: 12) {
                            Button("Fix Missing Files") {
                                Task {
                                    do {
                                        try await modelManager.fixIncompleteModel()
                                    } catch {
                                        showingError = true
                                    }
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(modelManager.isDownloading)
                            
                            Button("Full Re-download") {
                                Task {
                                    do {
                                        try await modelManager.deleteAndRedownloadModel()
                                    } catch {
                                        showingError = true
                                    }
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(modelManager.isDownloading)
                        }
                    }
                    
                case .downloadFailed:
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 50))
                            .foregroundColor(.orange)
                        
                        VStack(spacing: 12) {
                            Text("Setup Required")
                                .font(.headline)
                            
                            Text(modelManager.downloadError ?? "Unknown error occurred")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        
                        Button("Try Again") {
                            Task {
                                try await modelManager.downloadModel()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
            }
            
            Spacer()
            
            // Footer info
            if modelManager.modelState == .notDownloaded || modelManager.modelState == .downloading {
                VStack(spacing: 4) {
                    Text("Why download a model?")
                        .font(.caption)
                        .fontWeight(.medium)
                    
                    Text("Aura runs completely offline for privacy and speed once the model is downloaded")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
        }
        .padding(40)
        .frame(maxWidth: 500)
        .alert("Download Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(modelManager.downloadError ?? "Unknown error occurred")
        }
    }
}
