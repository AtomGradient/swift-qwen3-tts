import Foundation

enum TestResources {
    private static let fileManager = FileManager.default

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Qwen3TTSTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // swift-qwen3-tts
    }

    private static var workspaceRoot: URL {
        packageRoot.deletingLastPathComponent()
    }

    private static func existingPath(_ candidates: [String]) -> String? {
        for path in candidates where fileManager.fileExists(atPath: path) {
            return path
        }
        return nil
    }

    private static func resolve(
        env: String,
        defaults: [URL]
    ) -> String? {
        let envValue = ProcessInfo.processInfo.environment[env]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultPaths = defaults.map(\.path)
        let candidates = [envValue].compactMap { $0?.isEmpty == false ? $0 : nil } + defaultPaths
        return existingPath(candidates)
    }

    static var voiceDesignModelPath: String? {
        resolve(
            env: "QWEN3_TTS_VOICEDESIGN_MODEL_PATH",
            defaults: [
                workspaceRoot.appendingPathComponent("Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"),
                packageRoot.appendingPathComponent("Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"),
            ]
        )
    }

    static var baseModelPath: String? {
        resolve(
            env: "QWEN3_TTS_BASE_MODEL_PATH",
            defaults: [
                workspaceRoot.appendingPathComponent("Qwen3-TTS-12Hz-1.7B-Base-bf16"),
                packageRoot.appendingPathComponent("Qwen3-TTS-12Hz-1.7B-Base-bf16"),
            ]
        )
    }

    static var referenceAudioPath: String? {
        resolve(
            env: "QWEN3_TTS_REFERENCE_AUDIO_PATH",
            defaults: [
                workspaceRoot.appendingPathComponent("test_voice_clone.wav"),
                packageRoot.appendingPathComponent("test_voice_clone.wav"),
            ]
        )
    }

    static func skipMessage(_ name: String, env: String) -> String {
        "Skipping \(name): set \(env) or place test resources in expected relative paths."
    }
}
