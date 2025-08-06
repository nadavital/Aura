import Foundation

class ToolExecutor: ObservableObject {
    
    func executeShellCommand(_ command: String, in workingDirectory: String) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let pipe = Pipe()
            
            process.standardOutput = pipe
            process.standardError = pipe
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", command]
            process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
            
            do {
                try process.run()
                process.waitUntilExit()
                
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: data, encoding: .utf8) ?? ""
                
                if process.terminationStatus == 0 {
                    continuation.resume(returning: output)
                } else {
                    continuation.resume(throwing: ToolError.commandFailed(output))
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    func readFile(at path: String) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    let content = try String(contentsOfFile: path, encoding: .utf8)
                    continuation.resume(returning: content)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    func writeFile(content: String, to path: String) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    try content.write(toFile: path, atomically: true, encoding: .utf8)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    func listDirectory(at path: String) async throws -> [String] {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    let fileManager = FileManager.default
                    let contents = try fileManager.contentsOfDirectory(atPath: path)
                    continuation.resume(returning: contents.sorted())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    func createDirectory(at path: String) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    let fileManager = FileManager.default
                    try fileManager.createDirectory(atPath: path, withIntermediateDirectories: true)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

enum ToolError: Error, LocalizedError {
    case commandFailed(String)
    case fileNotFound
    case permissionDenied
    
    var errorDescription: String? {
        switch self {
        case .commandFailed(let output):
            return "Command failed: \(output)"
        case .fileNotFound:
            return "File not found"
        case .permissionDenied:
            return "Permission denied"
        }
    }
}