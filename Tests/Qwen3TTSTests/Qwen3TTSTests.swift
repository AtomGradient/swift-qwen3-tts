//
//  Qwen3TTSTests.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS unit tests
//

import Testing
import MLX
import Foundation

@testable import Qwen3TTS


// Run Qwen3 tests with:
//   swift test --filter Qwen3TTSTests


struct Qwen3TTSTests {

    /// Test SpeechTokenizer decode with fixed codes - detailed intermediate values
    /// Compare with Python debug_decode_detailed.py
    @Test func testSpeechTokenizerDecode() async throws {
        // 1. Load Qwen3 TTS model
        guard let modelPath = TestResources.voiceDesignModelPath else {
            print(TestResources.skipMessage("testSpeechTokenizerDecode", env: "QWEN3_TTS_VOICEDESIGN_MODEL_PATH"))
            return
        }
        print("\u{001B}[33mLoading Qwen3 TTS model for decode test...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("\u{001B}[32mQwen3 TTS model loaded!\u{001B}[0m")

        // 2. Use the exact same codes as Python debug_decode.py
        // Shape: [5, 16] - 5 time steps, 16 codebooks
        let codesData: [[Int32]] = [
            [1342, 313, 975, 826, 309, 933, 1642, 28, 782, 1965, 1680, 1507, 258, 1349, 828, 1102],
            [1014, 17, 2016, 285, 1712, 470, 543, 176, 1087, 1963, 588, 1860, 889, 1874, 1121, 1319],
            [1119, 1122, 594, 89, 770, 1644, 331, 242, 1183, 1676, 129, 96, 123, 1840, 661, 285],
            [1119, 1135, 215, 1377, 88, 1611, 904, 1274, 1895, 1872, 1246, 335, 1141, 320, 694, 242],
            [46, 1952, 1023, 1871, 596, 491, 757, 422, 692, 683, 651, 395, 1235, 1300, 618, 1498],
        ]

        // Convert to MLXArray: [batch=1, seq_len=5, num_quantizers=16]
        let flatCodes = codesData.flatMap { $0 }
        let codes = MLXArray(flatCodes).reshaped([1, 5, 16])
        print("codes shape: \(codes.shape)")

        guard let speechTokenizer = model.speechTokenizer else {
            throw NSError(domain: "Test", code: 1, userInfo: [NSLocalizedDescriptionKey: "speechTokenizer is nil"])
        }

        // Access decoder for detailed debugging
        let decoder = speechTokenizer.decoder

        // Step 1: Transpose codes [1, 5, 16] -> [1, 16, 5]
        let codesT = codes.transposed(0, 2, 1)
        print("\ncodes_t shape: \(codesT.shape)")

        // Step 2: quantizer.decode
        let quantized = decoder.quantizer.decode(codesT)
        eval(quantized)
        let qMin = MLX.min(quantized).item(Float.self)
        let qMax = MLX.max(quantized).item(Float.self)
        let qMean = MLX.mean(quantized).item(Float.self)
        let qStd = MLX.sqrt(MLX.mean(MLX.pow(quantized - MLXArray(qMean), 2))).item(Float.self)
        print("\n=== After quantizer.decode ===")
        print("quantized shape: \(quantized.shape)")  // Should be [1, 512, 5]
        print("quantized stats: min=\(String(format: "%.4f", qMin)), max=\(String(format: "%.4f", qMax)), mean=\(String(format: "%.6f", qMean)), std=\(String(format: "%.4f", qStd))")
        // Python: min=-81.5562, max=56.5420, std=12.8890
        let q_0_10_0 = quantized[0, 0..<10, 0]
        eval(q_0_10_0)
        print("quantized[0, :10, 0]: \(q_0_10_0)")
        // Python: [4.45, -4.80, 42.66, -24.35, -20.34, -6.54, -10.84, -9.93, 0.57, -6.90]

        // Step 3: pre_conv
        let preConvOut = decoder.preConv(quantized)
        eval(preConvOut)
        let pcMin = MLX.min(preConvOut).item(Float.self)
        let pcMax = MLX.max(preConvOut).item(Float.self)
        let pcStd = MLX.sqrt(MLX.mean(MLX.pow(preConvOut, 2))).item(Float.self)
        print("\n=== After pre_conv ===")
        print("pre_conv shape: \(preConvOut.shape)")  // Should be [1, 1024, 5]
        print("pre_conv stats: min=\(String(format: "%.4f", pcMin)), max=\(String(format: "%.4f", pcMax)), std=\(String(format: "%.4f", pcStd))")
        // Python: min=-1.4770, max=1.5426, std=0.1096

        // Step 4: pre_transformer
        let preT = preConvOut.transposed(0, 2, 1)  // [1, 5, 1024]
        let transformerOut = decoder.preTransformer(preT)
        let transformerOutT = transformerOut.transposed(0, 2, 1)  // [1, 1024, 5]
        eval(transformerOutT)
        let tfMin = MLX.min(transformerOutT).item(Float.self)
        let tfMax = MLX.max(transformerOutT).item(Float.self)
        let tfStd = MLX.sqrt(MLX.mean(MLX.pow(transformerOutT, 2))).item(Float.self)
        print("\n=== After pre_transformer ===")
        print("transformer shape: \(transformerOutT.shape)")
        print("transformer stats: min=\(String(format: "%.4f", tfMin)), max=\(String(format: "%.4f", tfMax)), std=\(String(format: "%.4f", tfStd))")
        // Python: min=-0.1119, max=0.1103, std=0.0185

        // Step 5: Upsample (4x = 2x * 2x)
        var upsampleOut = transformerOutT
        for (i, layers) in decoder.upsample.enumerated() {
            for layer in layers {
                if let conv = layer as? CausalTransposeConv1d {
                    upsampleOut = conv(upsampleOut)
                } else if let block = layer as? Qwen3TTS.ConvNeXtBlock {
                    upsampleOut = block(upsampleOut)
                }
            }
            eval(upsampleOut)
            let usMin = MLX.min(upsampleOut).item(Float.self)
            let usMax = MLX.max(upsampleOut).item(Float.self)
            let usStd = MLX.sqrt(MLX.mean(MLX.pow(upsampleOut, 2))).item(Float.self)
            print("\n=== After upsample block \(i) ===")
            print("shape: \(upsampleOut.shape)")
            print("stats: min=\(String(format: "%.4f", usMin)), max=\(String(format: "%.4f", usMax)), std=\(String(format: "%.4f", usStd))")
        }
        // Python: block 0: shape=(1, 1024, 10), std=0.2205
        // Python: block 1: shape=(1, 1024, 20), std=2.1252

        // Step 6: Main decoder layers
        var mainOut = upsampleOut
        // Access main decoder directly (no inner wrapper anymore)
        let mainDecoder = decoder.mainDecoder

        // Check initConv weights
        print("\n=== initConv weights ===")
        let initConvWeight = mainDecoder.initConv.conv.weight
        eval(initConvWeight)
        print("initConv.conv.weight shape: \(initConvWeight.shape)")
        // Python: (1536, 7, 1024) - MLX format [out, kernel, in]
        let wMin = MLX.min(initConvWeight).item(Float.self)
        let wMax = MLX.max(initConvWeight).item(Float.self)
        let wMean = MLX.mean(initConvWeight).item(Float.self)
        let wStd = MLX.sqrt(MLX.mean(MLX.pow(initConvWeight - MLXArray(wMean), 2))).item(Float.self)
        print("weight stats: min=\(String(format: "%.6f", wMin)), max=\(String(format: "%.6f", wMax)), mean=\(String(format: "%.6f", wMean)), std=\(String(format: "%.6f", wStd))")
        // Python: min=-0.057689, max=0.052053, std=0.001624

        if let initConvBias = mainDecoder.initConv.conv.bias {
            eval(initConvBias)
            print("initConv.conv.bias shape: \(initConvBias.shape)")
            let bMin = MLX.min(initConvBias).item(Float.self)
            let bMax = MLX.max(initConvBias).item(Float.self)
            let bMean = MLX.mean(initConvBias).item(Float.self)
            print("bias stats: min=\(String(format: "%.6f", bMin)), max=\(String(format: "%.6f", bMax)), mean=\(String(format: "%.6f", bMean))")
            // Python: min=-0.736545, max=0.121635, mean=-0.168909
        } else {
            print("initConv.conv.bias: nil")
        }

        // Test with ones input (same as Python)
        print("\n=== Test initConv with ones input ===")
        let testInput = MLXArray.ones([1, 1024, 20])
        let testOutput = mainDecoder.initConv(testInput)
        eval(testOutput)
        print("input shape: \(testInput.shape)")
        print("output shape: \(testOutput.shape)")
        let toMin = MLX.min(testOutput).item(Float.self)
        let toMax = MLX.max(testOutput).item(Float.self)
        let toMean = MLX.mean(testOutput).item(Float.self)
        let toStd = MLX.sqrt(MLX.mean(MLX.pow(testOutput - MLXArray(toMean), 2))).item(Float.self)
        print("output stats: min=\(String(format: "%.4f", toMin)), max=\(String(format: "%.4f", toMax)), mean=\(String(format: "%.6f", toMean)), std=\(String(format: "%.4f", toStd))")
        // Python: min=-0.8831, max=0.9967, mean=-0.161432, std=0.2207

        // Layer 0: initConv with actual data
        mainOut = mainDecoder.initConv(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 0 (initConv) ===")
        print("shape: \(mainOut.shape)")
        var mMin = MLX.min(mainOut).item(Float.self)
        var mMax = MLX.max(mainOut).item(Float.self)
        var mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 0: shape=(1, 1536, 20), std=0.5851

        // Check block0 SnakeBeta weights
        print("\n=== block0 SnakeBeta weights ===")
        let snake0Alpha = mainDecoder.block0.snake.alpha
        let snake0Beta = mainDecoder.block0.snake.beta
        eval(snake0Alpha, snake0Beta)
        let expAlpha = MLX.exp(snake0Alpha)
        let expBeta = MLX.exp(snake0Beta)
        eval(expAlpha, expBeta)
        print("alpha (exp): min=\(String(format: "%.6f", MLX.min(expAlpha).item(Float.self))), max=\(String(format: "%.6f", MLX.max(expAlpha).item(Float.self))), mean=\(String(format: "%.6f", MLX.mean(expAlpha).item(Float.self)))")
        print("beta (exp): min=\(String(format: "%.6f", MLX.min(expBeta).item(Float.self))), max=\(String(format: "%.6f", MLX.max(expBeta).item(Float.self))), mean=\(String(format: "%.6f", MLX.mean(expBeta).item(Float.self)))")
        // Python: alpha mean=0.819300, beta mean=0.961781

        // Layer 1: block0
        mainOut = mainDecoder.block0(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 1 (block0) ===")
        print("shape: \(mainOut.shape)")
        mMin = MLX.min(mainOut).item(Float.self)
        mMax = MLX.max(mainOut).item(Float.self)
        mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 1: shape=(1, 768, 160), std=0.5998

        // Layer 2: block1
        mainOut = mainDecoder.block1(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 2 (block1) ===")
        print("shape: \(mainOut.shape)")
        mMin = MLX.min(mainOut).item(Float.self)
        mMax = MLX.max(mainOut).item(Float.self)
        mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 2: shape=(1, 384, 800), std=0.4335

        // Layer 3: block2
        mainOut = mainDecoder.block2(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 3 (block2) ===")
        print("shape: \(mainOut.shape)")
        mMin = MLX.min(mainOut).item(Float.self)
        mMax = MLX.max(mainOut).item(Float.self)
        mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 3: shape=(1, 192, 3200), std=0.4099

        // Layer 4: block3
        mainOut = mainDecoder.block3(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 4 (block3) ===")
        print("shape: \(mainOut.shape)")
        mMin = MLX.min(mainOut).item(Float.self)
        mMax = MLX.max(mainOut).item(Float.self)
        mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 4: shape=(1, 96, 9600), std=8.2522

        // Layer 5: outSnake
        mainOut = mainDecoder.outSnake(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 5 (outSnake) ===")
        print("shape: \(mainOut.shape)")
        mMin = MLX.min(mainOut).item(Float.self)
        mMax = MLX.max(mainOut).item(Float.self)
        mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 5: shape=(1, 96, 9600), std=8.2532

        // Layer 6: outConv
        mainOut = mainDecoder.outConv(mainOut)
        eval(mainOut)
        print("\n=== After decoder layer 6 (outConv) ===")
        print("shape: \(mainOut.shape)")
        mMin = MLX.min(mainOut).item(Float.self)
        mMax = MLX.max(mainOut).item(Float.self)
        mStd = MLX.sqrt(MLX.mean(MLX.pow(mainOut, 2))).item(Float.self)
        print("stats: min=\(String(format: "%.4f", mMin)), max=\(String(format: "%.4f", mMax)), std=\(String(format: "%.4f", mStd))")
        // Python layer 6: shape=(1, 1, 9600), std=0.1712

        // Now call full decode and check final output
        print("\n=== Full decode ===")
        let (audio, audioLengths) = speechTokenizer.decode(codes)
        eval(audio, audioLengths)

        print("audio shape: \(audio.shape)")
        print("audioLengths: \(audioLengths)")

        let audioData = audio[0]
        eval(audioData)
        let audioMin = MLX.min(audioData).item(Float.self)
        let audioMax = MLX.max(audioData).item(Float.self)
        let audioMean = MLX.mean(audioData).item(Float.self)
        let audioStd = MLX.sqrt(MLX.mean(MLX.pow(audioData - MLXArray(audioMean), 2))).item(Float.self)

        print("audio stats: min=\(String(format: "%.4f", audioMin)), max=\(String(format: "%.4f", audioMax)), mean=\(String(format: "%.6f", audioMean)), std=\(String(format: "%.4f", audioStd))")
        // Python: min=-0.5261, max=0.5532, std=0.1712

        // Check quantizer output matches Python (critical first step)
        #expect(qStd > 10.0, "quantizer.decode std should be ~12.89 like Python")
        #expect(abs(qMin + 81.5) < 5.0, "quantizer.decode min should be ~-81.56 like Python")

        // Save audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift_fixed_codes.wav")
        try saveAudioArray(audio[0], sampleRate: 24000.0, to: outputURL)
        print("\u{001B}[32mSaved to: \(outputURL.path)\u{001B}[0m")
    }

    /// Test basic text-to-speech generation with Qwen3 TTS model (VoiceDesign mode)
    @Test func testQwen3TTSGenerate() async throws {
        // 1. Load Qwen3 TTS model from local path
        guard let modelPath = TestResources.voiceDesignModelPath else {
            print(TestResources.skipMessage("testQwen3TTSGenerate", env: "QWEN3_TTS_VOICEDESIGN_MODEL_PATH"))
            return
        }
        print("\u{001B}[33mLoading Qwen3 TTS model from: \(modelPath)...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("\u{001B}[32mQwen3 TTS model loaded!\u{001B}[0m")

        // 2. Generate audio from text with VoiceDesign mode
        let text = "Hello, this is a test of the Qwen3 text to speech model."
        let instruct = "A clear female voice with a warm and friendly tone."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")
        print("\u{001B}[33mVoice instruct: \"\(instruct)\"\u{001B}[0m")

        let audio = try await model.generate(
            text: text,
            instruct: instruct,
            language: "english",
            temperature: 0.9,
            topK: 50,
            topP: 1.0,
            repetitionPenalty: 1.05,
            maxTokens: 500
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tts_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with Qwen3 TTS model
    @Test func testQwen3TTSGenerateStream() async throws {
        // 1. Load Qwen3 TTS model from local path
        guard let modelPath = TestResources.voiceDesignModelPath else {
            print(TestResources.skipMessage("testQwen3TTSGenerateStream", env: "QWEN3_TTS_VOICEDESIGN_MODEL_PATH"))
            return
        }
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("\u{001B}[32mQwen3 TTS model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Qwen3 model."
        let instruct = "A clear male voice speaking at a moderate pace."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        var finalAudio: MLXArray?
        var generationInfo: Qwen3TTSGenerationInfo?

        for try await event in model.generateStream(
            text: text,
            instruct: instruct,
            language: "english",
            temperature: 0.9,
            topK: 50,
            maxTokens: 300
        ) {
            switch event {
            case .token(_):
                break  // No token-by-token events in current implementation
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen3_tts_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }

    /// Test Base model loading with encoder support (for voice cloning)
    @Test func testQwen3TTSBaseModelEncoder() async throws {
        // 1. Load Qwen3 TTS Base model (has encoder for voice cloning)
        guard let modelPath = TestResources.baseModelPath else {
            print(TestResources.skipMessage("testQwen3TTSBaseModelEncoder", env: "QWEN3_TTS_BASE_MODEL_PATH"))
            return
        }
        print("\u{001B}[33mLoading Qwen3 TTS Base model from: \(modelPath)...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("\u{001B}[32mQwen3 TTS Base model loaded!\u{001B}[0m")

        // 2. Check encoder is available
        #expect(model.speechTokenizer != nil, "Speech tokenizer should be loaded")
        #expect(model.speechTokenizer?.hasEncoder == true, "Base model should have encoder")
        #expect(model.supportsVoiceCloning == true, "Base model should support voice cloning")

        print("\u{001B}[32mEncoder check passed!\u{001B}[0m")

        // 3. Test encoder with dummy audio
        guard let speechTokenizer = model.speechTokenizer else {
            throw NSError(domain: "Test", code: 1, userInfo: [NSLocalizedDescriptionKey: "speechTokenizer is nil"])
        }

        // Create dummy audio: 1 second at 24kHz
        let dummyAudio = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 24000])
        print("\u{001B}[33mTesting encoder with dummy audio shape: \(dummyAudio.shape)\u{001B}[0m")

        let codes = try speechTokenizer.encode(dummyAudio)
        eval(codes)

        print("\u{001B}[32mEncoded codes shape: \(codes.shape)\u{001B}[0m")
        // Should be [1, 16, time_steps] where time_steps ~= 24000 / 1920 = 12.5 (12 or 13)
        #expect(codes.dim(0) == 1, "Batch size should be 1")
        #expect(codes.dim(1) == 16, "Should have 16 quantizers")
        #expect(codes.dim(2) > 0, "Should have time steps")

        let codesMin = MLX.min(codes).item(Int32.self)
        let codesMax = MLX.max(codes).item(Int32.self)
        print("Codes stats: min=\(codesMin), max=\(codesMax)")
        #expect(codesMin >= 0, "Code indices should be non-negative")
        #expect(codesMax < 2048, "Code indices should be less than codebook size")

        print("\u{001B}[32mBase model encoder test passed!\u{001B}[0m")
    }

    /// Test voice cloning with Base model
    @Test func testQwen3TTSVoiceClone() async throws {
        // 1. Load Qwen3 TTS Base model (supports voice cloning)
        guard let modelPath = TestResources.baseModelPath else {
            print(TestResources.skipMessage("testQwen3TTSVoiceClone", env: "QWEN3_TTS_BASE_MODEL_PATH"))
            return
        }
        print("\u{001B}[33mLoading Qwen3 TTS Base model for voice cloning...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("\u{001B}[32mQwen3 TTS Base model loaded!\u{001B}[0m")

        // 2. Check voice cloning is supported
        #expect(model.supportsVoiceCloning == true, "Base model should support voice cloning")

        // 3. Load reference audio
        guard let refAudioPath = TestResources.referenceAudioPath else {
            print(TestResources.skipMessage("testQwen3TTSVoiceClone", env: "QWEN3_TTS_REFERENCE_AUDIO_PATH"))
            return
        }
        print("\u{001B}[33mLoading reference audio from: \(refAudioPath)\u{001B}[0m")
        let (sampleRate, refAudioRaw) = try loadAudioArray(from: URL(fileURLWithPath: refAudioPath))
        print("  Sample rate: \(sampleRate), samples: \(refAudioRaw.shape[0])")

        // Ensure 24kHz (model requirement)
        #expect(sampleRate == 24000, "Reference audio should be 24kHz")
        let refAudio = refAudioRaw  // 1D array [samples]
        print("\u{001B}[32mReference audio loaded: \(refAudio.shape)\u{001B}[0m")

        // 4. Generate with voice cloning
        let text = "Hello, this is a voice cloning test using the Qwen3 TTS model."
        let refText = "This is the reference audio for voice cloning."  // Approximate transcription
        print("\u{001B}[33mGenerating audio with voice cloning...\u{001B}[0m")
        print("  Target text: \"\(text)\"")
        print("  Reference text: \"\(refText)\"")

        let audio = try model.generateVoiceClone(
            text: text,
            referenceAudio: refAudio,
            referenceText: refText,
            language: "english",
            temperature: 0.9,
            topK: 50,
            maxTokens: 500
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 5. Verify output
        #expect(audio.shape[0] > 0, "Generated audio should have samples")

        // 6. Save output
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_voice_clone_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved voice cloned audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test Chinese text generation
    @Test func testQwen3TTSChinese() async throws {
        // 1. Load model
        guard let modelPath = TestResources.voiceDesignModelPath else {
            print(TestResources.skipMessage("testQwen3TTSChinese", env: "QWEN3_TTS_VOICEDESIGN_MODEL_PATH"))
            return
        }
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("\u{001B}[32mQwen3 TTS model loaded!\u{001B}[0m")

        // 2. Generate Chinese audio
        let text = "你好，这是一个中文语音合成测试。"
        let instruct = "一个温柔的女声，语速适中。"
        print("\u{001B}[33mGenerating Chinese audio...\u{001B}[0m")

        let audio = try await model.generate(
            text: text,
            instruct: instruct,
            language: "chinese",
            temperature: 0.9,
            topK: 50,
            maxTokens: 500
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // Save
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tts_chinese_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved Chinese audio to\u{001B}[0m: \(outputURL.path)")
    }
}
