# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aura is a SwiftUI-based macOS application that provides AI-powered chat and coding assistance with local LLM integration. The app features a tabbed interface with three main sections: Chat, Code, and Settings.

## Development Commands

Since this is an Xcode project, use these commands for development:

**Build the project:**
```bash
xcodebuild -project Aura.xcodeproj -scheme Aura -configuration Debug build
```

**Clean build folder:**
```bash
xcodebuild -project Aura.xcodeproj -scheme Aura clean
```

**Run tests (if any exist):**
```bash
xcodebuild -project Aura.xcodeproj -scheme Aura test
```

**Open in Xcode:**
```bash
open Aura.xcodeproj
```

## Architecture

### Core Structure
- **AuraApp.swift**: Main app entry point with WindowGroup configuration for main window and quick lookup overlay
- **ContentView.swift**: Root view with TabView containing three main sections

### Models
- **Message.swift**: Core data models for chat functionality including `Message`, `ChatRequest`, `ChatResponse`, and related structs for LLM API communication

### Services
- **MLXModelManager.swift**: **NEW** - Handles MLX-based local AI inference with complete transformer implementation
- **LLMService.swift**: Handles communication with local LLM APIs (default: localhost:1234/v1), supports OpenAI-compatible endpoints
- **GlobalShortcutManager.swift**: Manages global keyboard shortcuts for quick lookup functionality  
- **ToolExecutor.swift**: Provides file system and shell command execution capabilities for the coding agent

### Views
- **ChatView.swift**: Traditional chat interface with message bubbles and conversation history
- **CodingAgentView.swift**: Specialized interface for coding tasks with working directory selection and tool execution
- **QuickLookupView.swift**: Overlay window for quick AI queries accessible via global shortcut
- **SettingsView.swift**: Configuration interface for LLM settings and feature toggles

## Key Features

### MLX Local AI Integration (NEW)
- **Bundled 20B Parameter Model**: Downloads gpt-oss-20b-MLX-8bit automatically (22GB)
- **Apple Silicon Optimization**: Native MLX framework for M1/M2/M3 GPU acceleration
- **Complete Transformer**: 24-layer GPT architecture with multi-head attention, MLP, embeddings
- **Safetensors Loading**: Parses 5 model shards with BF16/F16/F32 support
- **Real Tokenization**: Vocabulary loading and text encoding/decoding
- **Text Generation**: Autoregressive sampling with temperature control
- **No External Dependencies**: Self-contained AI inference

### Legacy LLM Integration
- Connects to local LLM servers (default localhost:1234/v1)
- Configurable base URL and model selection
- OpenAI-compatible API format

### Coding Agent
- File system operations (read/write/list/create directories)
- Shell command execution within selected working directory
- Integrated with LLM for code assistance

### Quick Lookup
- Global shortcut accessibility (⌘⇧Space)
- Overlay window for quick queries
- Independent from main chat interface

## Configuration

App settings are stored using `@AppStorage`:
- `llmBaseURL`: LLM server endpoint
- `llmModel`: Model identifier
- `quickLookupEnabled`: Toggle for quick lookup feature
- `globalShortcutEnabled`: Toggle for global keyboard shortcut

## MLX Implementation Details

### Current Status: ✅ COMPLETE
- **Safetensors Loading**: Parses all 5 model shards (model-00001 through model-00005.safetensors)
- **Transformer Architecture**: Complete GPTModel with 24 layers, MultiHeadAttention, MLP, LayerNorm
- **Tokenization**: Working vocabulary loading from tokenizer.json with encoding/decoding
- **Text Generation**: Autoregressive sampling with temperature control and EOS token handling
- **MLX Integration**: Proper Apple Silicon GPU acceleration with MLX framework

### Dependencies Added
```swift
// Package.swift dependencies:
MLX, MLXNN, MLXOptimizers, MLXRandom
```

### Key Classes
- **MLXModelManager**: Main orchestrator for model download, loading, and inference
- **SafetensorsLoader**: Parses safetensors format with BF16/F16/F32 conversion
- **Tokenizer**: Vocabulary management and text encoding/decoding
- **GPTModel**: Complete transformer with embeddings, attention, MLP, layer norms
- **MultiHeadAttention**: Scaled dot-product attention with Q/K/V projections
- **MLP**: Feed-forward network with GELU activation

### Model Details
- **Model**: lmstudio-community/gpt-oss-20b-MLX-8bit
- **Size**: ~22GB (5 safetensors shards + config files)
- **Architecture**: 24 layers, 16 attention heads, 4096 hidden size, 50257 vocab
- **Download**: Automatic via Hugging Face with retry logic and progress tracking

## Development Notes

- Uses modern SwiftUI with `@StateObject`, `@Published`, and async/await patterns
- Implements proper MainActor usage for UI updates
- File operations and shell commands run on background queues
- **NEW**: MLX framework integration for Apple Silicon GPU acceleration
- **NEW**: Complete safetensors parsing and transformer implementation
- Network entitlements configured for model downloads (sandbox + network client)