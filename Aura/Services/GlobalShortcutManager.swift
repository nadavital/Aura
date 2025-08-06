import SwiftUI

class GlobalShortcutManager: ObservableObject {
    @Published var isQuickLookupVisible = false
    
    func showQuickLookup() {
        isQuickLookupVisible = true
    }
    
    func hideQuickLookup() {
        isQuickLookupVisible = false
    }
}