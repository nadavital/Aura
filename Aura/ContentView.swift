//
//  ContentView.swift
//  Aura
//
//  Created by Avital, Nadav on 8/5/25.
//

import SwiftUI

struct ContentView: View {
    @State private var selectedTab: Int = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                ChatView()
            }
            .tabItem {
                Label("Chat", systemImage: "message")
            }
            .tag(0)
            
            NavigationStack {
                CodingAgentView()
            }
            .tabItem {
                Label("Code", systemImage: "terminal")
            }
            .tag(1)
            
            NavigationStack {
                SettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "gear")
            }
            .tag(2)
        }
    }
}

#Preview {
    ContentView()
}
