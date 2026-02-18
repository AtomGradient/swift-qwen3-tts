import Foundation
@preconcurrency import MLX

extension Qwen3TTSModel {
    /// Generate with streaming output.
    ///
    /// Emits `.token` events as first-codebook tokens are generated, then `.info` and final `.audio`.
    public func generateStream(
        text: String,
        speaker: String? = nil,
        instruct: String? = nil,
        language: String = "auto",
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        maxTokens: Int = 2048
    ) -> AsyncThrowingStream<Qwen3TTSGeneration, Error> {
        AsyncThrowingStream { continuation in
            Thread.detachNewThread {
                do {
                    let startTime = Date()
                    var generationTokenCount = 0

                    let onToken: (Int) -> Void = { token in
                        generationTokenCount += 1
                        continuation.yield(.token(token))
                    }

                    let audio: MLXArray
                    switch self.config.ttsModelType {
                    case "voice_design":
                        guard instruct != nil else {
                            throw Qwen3TTSError.invalidInput(
                                "VoiceDesign model requires 'instruct' to describe the voice " +
                                "(e.g., 'A cheerful young female voice with high pitch')"
                            )
                        }
                        audio = try self.generateVoiceDesign(
                            text: text,
                            language: language,
                            instruct: instruct,
                            temperature: temperature,
                            topK: topK,
                            topP: topP,
                            repetitionPenalty: repetitionPenalty,
                            maxTokens: maxTokens,
                            onToken: onToken
                        )

                    case "custom_voice":
                        guard let speaker = speaker else {
                            throw Qwen3TTSError.invalidInput(
                                "CustomVoice model requires 'speaker' (e.g., 'Vivian', 'Ryan'). " +
                                "Available speakers: \(self.supportedSpeakers.joined(separator: ", "))"
                            )
                        }
                        audio = try self.generateCustomVoice(
                            text: text,
                            speaker: speaker,
                            language: language,
                            instruct: instruct,
                            temperature: temperature,
                            topK: topK,
                            topP: topP,
                            repetitionPenalty: repetitionPenalty,
                            maxTokens: maxTokens,
                            onToken: onToken
                        )

                    case "base":
                        guard let speaker = speaker else {
                            throw Qwen3TTSError.invalidInput(
                                "Base model requires 'speaker' (e.g., 'Vivian', 'Ryan'). " +
                                "Available speakers: \(self.supportedSpeakers.joined(separator: ", "))"
                            )
                        }
                        audio = try self.generateCustomVoice(
                            text: text,
                            speaker: speaker,
                            language: language,
                            instruct: nil,
                            temperature: temperature,
                            topK: topK,
                            topP: topP,
                            repetitionPenalty: repetitionPenalty,
                            maxTokens: maxTokens,
                            onToken: onToken
                        )

                    default:
                        audio = try self.generateVoiceDesign(
                            text: text,
                            language: language,
                            instruct: instruct,
                            temperature: temperature,
                            topK: topK,
                            topP: topP,
                            repetitionPenalty: repetitionPenalty,
                            maxTokens: maxTokens,
                            onToken: onToken
                        )
                    }

                    let totalTime = Date().timeIntervalSince(startTime)
                    let promptTokenCount = self.tokenizer?.encode(text: text).count ?? 0
                    let tokensPerSecond = totalTime > 0 ? Double(generationTokenCount) / totalTime : 0

                    let info = Qwen3TTSGenerationInfo(
                        promptTokenCount: promptTokenCount,
                        generationTokenCount: generationTokenCount,
                        prefillTime: 0,
                        generateTime: totalTime,
                        tokensPerSecond: tokensPerSecond,
                        peakMemoryUsage: Double(GPU.peakMemory) / 1e9
                    )

                    continuation.yield(.info(info))
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
