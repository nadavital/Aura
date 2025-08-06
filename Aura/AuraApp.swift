//
//  AuraApp.swift
//  Aura
//
//  Created by Avital, Nadav on 8/5/25.
//

import SwiftUI

@main
struct AuraApp: App {
    @StateObject private var shortcutManager = GlobalShortcutManager()
    @StateObject private var modelManager = MLXModelManager()
    
    var body: some Scene {
        WindowGroup {
            MainAppView()
                .environmentObject(shortcutManager)
                .environmentObject(modelManager)
        }
        
        // Quick Lookup Window
        WindowGroup("Quick Lookup", id: "quick-lookup") {
            QuickLookupView()
                .environmentObject(shortcutManager)
                .environmentObject(modelManager)
        }
        .windowStyle(.plain)
        .windowResizability(.contentSize)
    }
}
