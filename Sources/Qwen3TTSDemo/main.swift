//
//  main.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS Demo - Command line tool to generate speech from text
//
//  Usage:
//    swift run Qwen3TTSDemo --text "Hello world" --instruct "A clear female voice"
//

import Foundation
import MLX
import Qwen3TTS

// MARK: - Arguments

struct Arguments {
    var text: String = "Hello world"
    var instruct: String? = nil
    var speaker: String? = nil
    var modelPath: String = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    var output: String = "output.wav"
    var language: String = "auto"
    var temperature: Float = 0.9
    var topK: Int = 50
    var maxTokens: Int = 2048
    var referenceAudio: String? = nil
    var referenceText: String? = nil
    var profile: Bool = false
    var profileOutput: String = "activation_profile.json"

    static func parse() -> Arguments {
        var args = Arguments()
        let arguments = CommandLine.arguments

        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--text", "-t":
                i += 1
                if i < arguments.count { args.text = arguments[i] }
            case "--instruct", "-i":
                i += 1
                if i < arguments.count { args.instruct = arguments[i] }
            case "--speaker", "-s":
                i += 1
                if i < arguments.count { args.speaker = arguments[i] }
            case "--model", "-m":
                i += 1
                if i < arguments.count { args.modelPath = arguments[i] }
            case "--output", "-o":
                i += 1
                if i < arguments.count { args.output = arguments[i] }
            case "--language", "-l":
                i += 1
                if i < arguments.count { args.language = arguments[i] }
            case "--temperature":
                i += 1
                if i < arguments.count { args.temperature = Float(arguments[i]) ?? 0.9 }
            case "--top-k":
                i += 1
                if i < arguments.count { args.topK = Int(arguments[i]) ?? 50 }
            case "--max-tokens":
                i += 1
                if i < arguments.count { args.maxTokens = Int(arguments[i]) ?? 2048 }
            case "--reference-audio":
                i += 1
                if i < arguments.count { args.referenceAudio = arguments[i] }
            case "--reference-text":
                i += 1
                if i < arguments.count { args.referenceText = arguments[i] }
            case "--profile":
                args.profile = true
            case "--profile-output":
                i += 1
                if i < arguments.count { args.profileOutput = arguments[i] }
            case "--help", "-h":
                printHelp()
                exit(0)
            default:
                break
            }
            i += 1
        }

        return args
    }

    static func printHelp() {
        print("""
        Qwen3 TTS Demo - Text to Speech Generation

        Usage:
          swift run Qwen3TTSDemo [options]

        Options:
          --text, -t           Text to synthesize (required)
          --instruct, -i       Voice style description (for VoiceDesign/CustomVoice)
          --speaker, -s        Speaker name (for CustomVoice: Vivian, Ryan, Aiden, etc.)
          --model, -m          Local path or HuggingFace repo ID (e.g. "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
          --output, -o         Output WAV file path (default: "output.wav")
          --language, -l       Language: auto, zh, en, ja, ko, etc. (default: "auto")
          --temperature        Sampling temperature (default: 0.9)
          --top-k              Top-K sampling (default: 50)
          --max-tokens         Maximum tokens to generate (default: 2048)
          --reference-audio    Reference audio path for voice cloning (Base model only)
          --reference-text     Reference audio transcript for voice cloning
          --profile            Run activation profiling (records per-neuron FFN activations)
          --profile-output     Path for activation profile JSON (default: "activation_profile.json")
          --help, -h           Show this help message

        Models:
          VoiceDesign  - Create any voice from text description (--instruct)
          CustomVoice  - Use predefined speakers (--speaker) with emotion control (--instruct)
          Base         - Voice cloning (--reference-audio + --reference-text)

        Examples:
          # VoiceDesign - create voice from description
          swift run Qwen3TTSDemo --model /path/to/VoiceDesign --text "Hello" --instruct "A warm female voice"

          # CustomVoice - use predefined speaker with emotion
          swift run Qwen3TTSDemo --model /path/to/CustomVoice --text "Hello" --speaker Ryan --instruct "Happy"

          # Voice Cloning - clone a voice from reference audio
          swift run Qwen3TTSDemo --model /path/to/Base --text "New content" --reference-audio ref.wav --reference-text "Reference transcript"
        """)
    }
}

// MARK: - WAV Writer

func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()

    // RIFF header
    data.append(contentsOf: "RIFF".utf8)
    let fileSize = UInt32(36 + samples.count * 2)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)

    // fmt chunk
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // mono
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })  // block align
    data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) }) // bits per sample

    // data chunk
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(samples.count * 2).littleEndian) { Array($0) })

    // Audio samples
    for sample in samples {
        let clamped = max(-1.0, min(1.0, sample))
        let int16 = Int16(clamped * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
    }

    try data.write(to: url)
}

// MARK: - Main

// MARK: - Profiling Texts

/// Diverse texts for activation profiling — covers different phonemes, lengths, and styles
let profilingTexts: [(text: String, language: String)] = [
    // English — various styles and lengths
    ("Hello, my name is Aiden. Nice to meet you.", "en"),
    ("The quick brown fox jumps over the lazy dog.", "en"),
    ("Please remember to take your medication at eight o'clock tonight.", "en"),
    ("Wow, that's absolutely incredible! I can't believe it!", "en"),
    ("The weather forecast shows heavy rain tomorrow with temperatures dropping to forty degrees.", "en"),
    ("One, two, three, four, five, six, seven, eight, nine, ten.", "en"),
    ("I'm sorry to hear that. Is there anything I can do to help?", "en"),
    ("According to the latest research, artificial intelligence is transforming healthcare.", "en"),
    ("Good morning! How did you sleep last night?", "en"),
    ("The restaurant is located at three hundred and twenty five Main Street.", "en"),
    // Chinese — various styles
    ("你好，我是你的智能助手，有什么可以帮助你的吗？", "chinese"),
    ("今天天气真不错，适合出去散步。", "chinese"),
    ("请注意，前方路口即将变为红灯，请减速慢行。", "chinese"),
    ("根据最新的研究报告，人工智能技术正在快速发展。", "chinese"),
    ("一二三四五六七八九十，百千万。", "chinese"),
    // Japanese
    ("こんにちは、今日はいい天気ですね。", "japanese"),
    ("東京の桜がとても綺麗です。", "japanese"),
    // Korean
    ("안녕하세요, 만나서 반갑습니다.", "korean"),
    // More English with different patterns
    ("Shh, be very quiet. The baby is sleeping.", "en"),
    ("BREAKING NEWS: Scientists discover new species in the deep ocean!", "en"),
]

@main
struct Qwen3TTSDemo {
    static func main() async throws {
        let args = Arguments.parse()

        if args.profile {
            try await runProfiling(args)
        } else {
            try await runGeneration(args)
        }
    }

    static func runGeneration(_ args: Arguments) async throws {
        print("Qwen3 TTS Demo")
        print("==============")
        print("Text: \"\(args.text)\"")
        if let speaker = args.speaker {
            print("Speaker: \(speaker)")
        }
        if let instruct = args.instruct {
            print("Instruct: \"\(instruct)\"")
        }
        if let refAudio = args.referenceAudio {
            print("Reference Audio: \(refAudio)")
        }
        if let refText = args.referenceText {
            print("Reference Text: \"\(refText)\"")
        }
        print("Model: \(args.modelPath)")
        print("Output: \(args.output)")
        print()

        // Load model
        print("Loading model...")
        let startLoad = Date()
        let model = try await Qwen3TTSModel.fromPretrained(args.modelPath)
        let loadTime = Date().timeIntervalSince(startLoad)
        print("Model loaded in \(String(format: "%.2f", loadTime))s")
        print()

        // Generate audio
        print("Generating audio...")
        let startGen = Date()

        let audio: MLXArray

        // Check if voice cloning mode
        if let refAudioPath = args.referenceAudio, let refText = args.referenceText {
            // Voice Cloning mode
            guard model.supportsVoiceCloning else {
                print("Error: This model doesn't support voice cloning. Use a Base model.")
                exit(1)
            }

            print("Mode: Voice Cloning")

            // Load reference audio
            let refURL = URL(fileURLWithPath: refAudioPath)
            let (sampleRate, refAudioRaw) = try loadAudioArray(from: refURL)

            // Check sample rate
            if sampleRate != 24000 {
                print("Warning: Reference audio is \(sampleRate)Hz, expected 24000Hz. Results may vary.")
            }
            let refAudio = refAudioRaw

            audio = try model.generateVoiceClone(
                text: args.text,
                referenceAudio: refAudio,
                referenceText: refText,
                language: args.language,
                temperature: args.temperature,
                topK: args.topK,
                maxTokens: args.maxTokens
            )
        } else {
            // VoiceDesign or CustomVoice mode
            audio = try await model.generate(
                text: args.text,
                speaker: args.speaker,
                instruct: args.instruct,
                language: args.language,
                temperature: args.temperature,
                topK: args.topK,
                maxTokens: args.maxTokens
            )
        }

        let genTime = Date().timeIntervalSince(startGen)

        // Convert to Float array
        eval(audio)
        let samples = audio.asArray(Float.self)
        let duration = Float(samples.count) / Float(model.sampleRate)

        print("Generated \(samples.count) samples (\(String(format: "%.2f", duration))s audio)")
        print("Generation time: \(String(format: "%.2f", genTime))s")
        print("Real-time factor: \(String(format: "%.2fx", Double(duration) / genTime))")
        print()

        // Save to WAV
        let outputURL = URL(fileURLWithPath: args.output)
        try writeWAV(samples, sampleRate: model.sampleRate, to: outputURL)
        print("Saved to: \(outputURL.path)")

        // Show peak memory
        let peakMem = Double(GPU.peakMemory) / 1e9
        print("Peak memory: \(String(format: "%.2f", peakMem)) GB")
    }

    static func runProfiling(_ args: Arguments) async throws {
        let speaker = args.speaker ?? "Aiden"

        print("=== Qwen3 TTS Activation Profiler ===")
        print("Speaker: \(speaker)")
        print("Model: \(args.modelPath)")
        print("Texts: \(profilingTexts.count) diverse inputs")
        print()

        // Load model
        print("Loading model...")
        let startLoad = Date()
        let model = try await Qwen3TTSModel.fromPretrained(args.modelPath)
        let loadTime = Date().timeIntervalSince(startLoad)
        print("Model loaded in \(String(format: "%.2f", loadTime))s")

        // Get model dimensions from config
        guard let talkerConfig = model.config.talkerConfig else {
            print("Error: No talker config found")
            exit(1)
        }
        let numLayers = talkerConfig.numHiddenLayers
        let intermediateSize = talkerConfig.intermediateSize

        print("Architecture: \(numLayers) layers, intermediate_size=\(intermediateSize)")
        print()

        // Enable profiling
        let profiler = ActivationProfiler.shared
        profiler.enable(numLayers: numLayers, intermediateSize: intermediateSize)

        // Run through all profiling texts
        for (index, item) in profilingTexts.enumerated() {
            print("  [\(index + 1)/\(profilingTexts.count)] \"\(item.text.prefix(50))...\" (lang=\(item.language))")

            do {
                let audio = try await model.generate(
                    text: item.text,
                    speaker: speaker,
                    language: item.language,
                    temperature: args.temperature,
                    topK: args.topK,
                    maxTokens: args.maxTokens
                )

                eval(audio)
                let samples = audio.asArray(Float.self)
                let duration = Float(samples.count) / Float(model.sampleRate)
                print("    -> \(String(format: "%.1f", duration))s audio, \(samples.count) samples")

                profiler.flushRun()

                // Clear GPU cache between runs to avoid memory buildup
                GPU.clearCache()
            } catch {
                print("    -> ERROR: \(error)")
            }
        }

        print()
        print("Profiling complete!")

        // Print summary
        profiler.printSummary()

        // Save results
        try profiler.saveJSON(to: args.profileOutput)

        // Show peak memory
        let peakMem = Double(GPU.peakMemory) / 1e9
        print("Peak memory: \(String(format: "%.2f", peakMem)) GB")
    }
}
