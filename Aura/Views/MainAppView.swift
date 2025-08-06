import SwiftUI

struct MainAppView: View {
    @EnvironmentObject var modelManager: MLXModelManager
    
    var body: some View {
        Group {
            switch modelManager.modelState {
            case .checking:
                // Show loading while checking for model
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                    Text("Checking model status...")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                
            case .notDownloaded, .downloading, .corrupted, .downloadFailed:
                // Show model setup
                ModelSetupView(modelManager: modelManager)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                
            case .ready:
                // Show main app interface
                ContentView()
            }
        }
        .animation(.easeInOut(duration: 0.3), value: modelManager.modelState)
    }
}