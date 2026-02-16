//
//  Qwen3.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS Main Model
//  Text-to-Speech using VoiceDesign mode
//
//  Ported from Python mlx-audio implementation
//

import Foundation
@preconcurrency import MLX
import HuggingFace
import Tokenizers
import MLXLMCommon
import MLXFast
import MLXNN

// MARK: - Type Aliases

public typealias Qwen3TTSError = AudioGenerationError
public typealias Qwen3TTSGenerationInfo = AudioGenerationInfo
public typealias Qwen3TTSGeneration = AudioGeneration

// MARK: - Qwen3 TTS Model

/// Main Qwen3 TTS Model for text-to-speech generation
public class Qwen3TTSModel: Module {
    public let config: Qwen3TTSModelConfig

    /// Main Talker model
    let talker: Qwen3TTSTalkerForConditionalGeneration

    /// Speech tokenizer (decoder only for VoiceDesign mode)
    var speechTokenizer: Qwen3TTSSpeechTokenizer?

    /// Speaker encoder for voice cloning (ECAPA-TDNN)
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: Qwen3TTSSpeakerEncoder?

    /// Text tokenizer (HuggingFace)
    public var tokenizer: Tokenizer?

    public init(_ config: Qwen3TTSModelConfig) {
        self.config = config

        guard let talkerConfig = config.talkerConfig else {
            fatalError("Talker config is required")
        }

        self.talker = Qwen3TTSTalkerForConditionalGeneration(talkerConfig)

        // Initialize speaker encoder if config is available
        if let speakerEncoderConfig = config.speakerEncoderConfig {
            self._speakerEncoder.wrappedValue = Qwen3TTSSpeakerEncoder(speakerEncoderConfig)
        }
    }

    /// Check if voice cloning is available (requires speaker encoder)
    public var hasVoiceCloning: Bool {
        return speakerEncoder != nil
    }

    // MARK: - Token Sampling

    /// Apply top-k filtering to logits (matching Python mlx_lm.sample_utils.apply_top_k)
    private func applyTopK(_ logits: MLXArray, _ topK: Int) -> MLXArray {
        let vocabSize = logits.dim(-1)
        guard topK > 0 && topK < vocabSize else { return logits }

        // Use argPartition to find indices of tokens NOT in top-k
        // argPartition(-logits, kth=k-1) puts the k largest values in positions 0..k-1
        let negLogits = -logits
        let partitioned = argPartition(negLogits, kth: topK - 1, axis: -1)

        // Indices after position k-1 should be masked
        let maskIdx = partitioned[0..., topK...]

        // Set those positions to -inf
        let maskedLogits = putAlong(
            logits,
            maskIdx,
            values: MLXArray(-Float.infinity),
            axis: -1
        )

        return maskedLogits
    }

    /// Apply top-p (nucleus) filtering to logits (matching Python mlx_lm.sample_utils.apply_top_p)
    private func applyTopP(_ logits: MLXArray, _ topP: Float) -> MLXArray {
        guard topP > 0 && topP < 1.0 else { return logits }

        // Convert to probabilities
        let probs = exp(logits)

        // Sort in ascending order
        let sortedIndices = argSort(logits, axis: -1)
        let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

        // Cumulative sum
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Rearrange back to original order
        let vocabSize = logits.dim(-1)
        let inverseIndices = putAlong(
            MLXArray.zeros(like: sortedIndices),
            sortedIndices,
            values: MLXArray(Int32(0)..<Int32(vocabSize)).expandedDimensions(axis: 0),
            axis: -1
        )
        let cumulativeProbsOriginal = takeAlong(cumulativeProbs, inverseIndices, axis: -1)

        // Select tokens with cumulative probs above threshold (1 - top_p)
        return MLX.where(cumulativeProbsOriginal .> (1.0 - topP), logits, MLXArray(-Float.infinity))
    }

    /// Categorical sampling with temperature (matching Python mlx_lm.sample_utils.categorical_sampling)
    private func categoricalSampling(_ logits: MLXArray, temperature: Float) -> MLXArray {
        // Python: mx.random.categorical(logits * (1 / temp))
        // CRITICAL: Force evaluation before scaling to ensure all logit modifications are applied
        eval(logits)
        let scaledLogits = logits * (1.0 / temperature)
        return categorical(scaledLogits)
    }

    /// Sample next token from logits using categorical sampling
    /// Matches Python mlx_lm.sample_utils implementation exactly
    private func sampleToken(
        _ logits: MLXArray,
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        generatedTokens: [Int] = [],
        suppressTokens: [Int]? = nil,
        eosTokenId: Int? = nil
    ) -> MLXArray {
        // Get last position logits, keeping batch dimension [1, vocab_size]
        var logitsProcessed: MLXArray
        if logits.ndim == 3 {
            logitsProcessed = logits[0..., -1, 0...]  // [1, vocab_size]
        } else if logits.ndim == 2 {
            logitsProcessed = logits  // Already [1, vocab_size]
        } else {
            logitsProcessed = logits.expandedDimensions(axis: 0)  // [1, vocab_size]
        }

        let vocabSize = logitsProcessed.dim(-1)

        // 1. Suppress invalid tokens (set to -inf)
        if let suppress = suppressTokens, !suppress.isEmpty {
            let suppressIdx = MLXArray(suppress.map { Int32($0) }).expandedDimensions(axis: 0)
            logitsProcessed = putAlong(
                logitsProcessed,
                suppressIdx,
                values: MLXArray(-Float.infinity),
                axis: -1
            )
        }

        // 2. Apply repetition penalty
        if !generatedTokens.isEmpty && repetitionPenalty != 1.0 {
            let uniqueTokens = Array(Set(generatedTokens)).filter { $0 < vocabSize }
            if !uniqueTokens.isEmpty {
                let tokenIds = MLXArray(uniqueTokens.map { Int32($0) }).expandedDimensions(axis: 0)
                let selectedLogits = takeAlong(logitsProcessed, tokenIds, axis: -1)

                // Apply penalty: multiply if negative, divide if positive
                let penalized = MLX.where(
                    selectedLogits .< 0,
                    selectedLogits * repetitionPenalty,
                    selectedLogits / repetitionPenalty
                )

                logitsProcessed = putAlong(logitsProcessed, tokenIds, values: penalized, axis: -1)
            }
        }

        // 3. Greedy decoding if temperature is 0
        if temperature <= 0 {
            let token = argMax(logitsProcessed, axis: -1, keepDims: true)
            return token
        }

        // 4. Save EOS logit (before any filtering!)
        var eosLogit: MLXArray? = nil
        if let eos = eosTokenId, eos < vocabSize {
            eosLogit = logitsProcessed[0..., eos..<(eos + 1)]
        }

        // 5. Apply top-k (BEFORE temperature, matching Python!)
        if topK > 0 && topK < vocabSize {
            logitsProcessed = applyTopK(logitsProcessed, topK)
        }

        // 6. Apply top-p (optional)
        if topP > 0.0 && topP < 1.0 {
            logitsProcessed = applyTopP(logitsProcessed, topP)
        }

        // 7. Restore EOS logit (original value, not scaled by temperature)
        if let eos = eosTokenId, let savedEosLogit = eosLogit, eos < vocabSize {
            let eosIdx = MLXArray([Int32(eos)]).expandedDimensions(axis: 0)
            logitsProcessed = putAlong(logitsProcessed, eosIdx, values: savedEosLogit, axis: -1)
        }

        // 8. Categorical sampling with temperature (temperature applied inside)
        let sampledToken = categoricalSampling(logitsProcessed, temperature: temperature)

        return sampledToken.expandedDimensions(axis: 0)  // [1, 1]
    }

    // MARK: - Speaker Embedding

    /// Extract speaker embedding from reference audio for voice cloning
    /// - Parameters:
    ///   - audio: Audio waveform [samples]
    ///   - sampleRate: Audio sample rate (must be 24000)
    /// - Returns: Speaker embedding [1, enc_dim]
    public func extractSpeakerEmbedding(_ audio: MLXArray, sampleRate: Int = 24000) throws -> MLXArray {
        guard sampleRate == 24000 else {
            throw Qwen3TTSError.invalidInput("Only 24kHz audio is supported for speaker embedding extraction")
        }

        guard let encoder = speakerEncoder else {
            throw Qwen3TTSError.modelNotInitialized("Speaker encoder not available for this model")
        }

        // Compute mel spectrogram
        let mels = melSpectrogram(
            audio,
            nFft: 1024,
            numMels: 128,
            sampleRate: 24000,
            hopSize: 256,
            winSize: 1024,
            fMin: 0,
            fMax: 12000
        )
        eval(mels)

        // Extract embedding
        let speakerEmbedding = encoder(mels)
        eval(speakerEmbedding)

        return speakerEmbedding
    }

    // MARK: - Input Preparation

    /// Prepare generation inputs for VoiceDesign/CustomVoice mode
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - language: Language code (auto, chinese, english, etc.)
    ///   - speaker: Speaker name for CustomVoice mode (e.g., "Vivian", "Ryan")
    ///   - instruct: Voice description (VoiceDesign) or emotion/style instruction (CustomVoice)
    private func prepareGenerationInputs(
        text: String,
        language: String = "auto",
        speaker: String? = nil,
        instruct: String? = nil
    ) -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        guard let tokenizer = self.tokenizer else {
            fatalError("Tokenizer not loaded")
        }

        guard let talkerConfig = config.talkerConfig else {
            fatalError("Talker config not available")
        }

        // 1. Tokenize text
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).expandedDimensions(axis: 0)

        // 2. Get text embeddings
        let textEmbedRaw = talker.embedText(inputIds)
        let textEmbed = talker.textProjection(textEmbedRaw)

        // 3. TTS special token embeddings
        let ttsTokens = MLXArray([
            Int32(config.ttsBosTokenId),
            Int32(config.ttsEosTokenId),
            Int32(config.ttsPadTokenId)
        ]).expandedDimensions(axis: 0)
        let ttsEmbedsRaw = talker.embedText(ttsTokens)
        let ttsEmbeds = talker.textProjection(ttsEmbedsRaw)

        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        // 4. Speaker embedding (for CustomVoice/Base models)
        var speakerEmbed: MLXArray? = nil
        if let speaker = speaker,
           let spkIdMap = talkerConfig.spkId,
           let spkTokenId = spkIdMap[speaker.lowercased()] {
            let spkIds = MLXArray([Int32(spkTokenId)]).expandedDimensions(axis: 0)
            speakerEmbed = talker.getInputEmbeddings()(spkIds)
        }

        // 5. Get language ID (check for dialect override)
        var languageId: Int? = nil
        let langLower = language.lowercased()
        if langLower != "auto" {
            languageId = talkerConfig.codecLanguageId[langLower]
        }

        // Check for dialect override (e.g., Eric -> sichuan_dialect)
        if (langLower == "chinese" || langLower == "auto"),
           let speaker = speaker,
           let spkIsDialect = talkerConfig.spkIsDialect,
           let dialectValue = spkIsDialect[speaker.lowercased()],
           let dialectName = dialectValue.dialectName {
            if let dialectId = talkerConfig.codecLanguageId[dialectName] {
                languageId = dialectId
            }
        }

        // 6. Build codec prefix
        var codecPrefill: [Int32]
        if languageId == nil {
            // No language: [nothink, think_bos, think_eos]
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            // With language: [think, think_bos, language_id, think_eos]
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(languageId!),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        let codecPrefillArray = MLXArray(codecPrefill).expandedDimensions(axis: 0)
        var codecEmbed = talker.getInputEmbeddings()(codecPrefillArray)

        // Add speaker embedding if available, then [pad, bos] suffix
        let codecSuffixArray = MLXArray([
            Int32(talkerConfig.codecPadId),
            Int32(talkerConfig.codecBosId)
        ]).expandedDimensions(axis: 0)
        let codecEmbedSuffix = talker.getInputEmbeddings()(codecSuffixArray)

        if let speakerEmbed = speakerEmbed {
            // Insert speaker embedding: [prefix..., speaker, pad, bos]
            codecEmbed = MLX.concatenated([
                codecEmbed,
                speakerEmbed.reshaped([1, 1, -1]),
                codecEmbedSuffix
            ], axis: 1)
        } else {
            codecEmbed = MLX.concatenated([codecEmbed, codecEmbedSuffix], axis: 1)
        }

        // 7. Instruct embedding (VoiceDesign/CustomVoice mode)
        var instructEmbed: MLXArray? = nil
        if let instruct = instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).expandedDimensions(axis: 0)
            let instructEmbedRaw = talker.embedText(instructIds)
            instructEmbed = talker.textProjection(instructEmbedRaw)
        }

        // 8. Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = textEmbed[0..., 0..<3, 0...]

        // 9. Build combined embedding
        // tts_pad * (codec_len - 2) + tts_bos
        let codecLen = codecEmbed.dim(1)
        let padCount = codecLen - 2
        let padEmbeds = MLX.broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(2)])
        var combinedEmbed = MLX.concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., 0..<(codecLen - 1), 0...]

        // 10. Complete input embeds
        var inputEmbeds: MLXArray
        if let instructEmbed = instructEmbed {
            inputEmbeds = MLX.concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = MLX.concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // 11. Add first text token
        let firstTextEmbed = textEmbed[0..., 3..<4, 0...] + codecEmbed[0..., (codecLen - 1)..., 0...]
        inputEmbeds = MLX.concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        // 12. Trailing text (remaining text + EOS)
        let textLen = textEmbed.dim(1)
        let trailingStart = 4
        let trailingEnd = textLen - 5  // Exclude last 5 tokens

        var trailingTextHidden: MLXArray
        if trailingEnd > trailingStart {
            trailingTextHidden = MLX.concatenated([
                textEmbed[0..., trailingStart..<trailingEnd, 0...],
                ttsEosEmbed
            ], axis: 1)
        } else {
            trailingTextHidden = ttsEosEmbed
        }

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    /// Prepare generation inputs for ICL (In-Context Learning) voice cloning
    /// - Parameters:
    ///   - text: Target text to synthesize
    ///   - refAudio: Reference audio waveform [samples] or [1, 1, samples]
    ///   - refText: Transcript of the reference audio
    ///   - language: Language code
    /// - Returns: (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes)
    private func prepareICLGenerationInputs(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String = "auto"
    ) throws -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray, refCodes: MLXArray) {
        guard let tokenizer = self.tokenizer else {
            throw Qwen3TTSError.modelNotInitialized("Tokenizer not loaded")
        }

        guard let talkerConfig = config.talkerConfig else {
            throw Qwen3TTSError.modelNotInitialized("Talker config not available")
        }

        guard let speechTokenizer = speechTokenizer, speechTokenizer.hasEncoder else {
            throw Qwen3TTSError.modelNotInitialized("Speech tokenizer encoder not available")
        }

        // 1. Encode reference audio -> ref_codes [1, 16, ref_time]
        var audioForEncode = refAudio
        if refAudio.ndim == 1 {
            audioForEncode = refAudio.expandedDimensions(axes: [0, 1])  // [1, 1, samples]
        } else if refAudio.ndim == 2 {
            audioForEncode = refAudio.expandedDimensions(axis: 0)  // [1, 1, samples]
        }
        let refCodes = try speechTokenizer.encode(audioForEncode)  // [1, 16, ref_time]
        eval(refCodes)

        // 2. Tokenize ref_text and target_text separately
        // ref_text format: <|im_start|>assistant\n{ref_text}<|im_end|>\n
        let refChat = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(text: refChat).map { Int32($0) }).expandedDimensions(axis: 0)
        // Pure ref text tokens: skip first 3 (role) and last 2 (<|im_end|>\n)
        let refTextIds = refIds[0..., 3..<(refIds.dim(1) - 2)]

        // target_text format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let targetChat = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChat).map { Int32($0) }).expandedDimensions(axis: 0)
        // Pure target text tokens: skip first 3 (role) and last 5 (trailing template)
        let textIds = targetIds[0..., 3..<(targetIds.dim(1) - 5)]

        // 3. TTS special tokens
        let ttsTokens = MLXArray([
            Int32(config.ttsBosTokenId),
            Int32(config.ttsEosTokenId),
            Int32(config.ttsPadTokenId)
        ]).expandedDimensions(axis: 0)
        let ttsEmbedsRaw = talker.embedText(ttsTokens)
        let ttsEmbeds = talker.textProjection(ttsEmbedsRaw)

        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        // 4. Build text_embed: text_projection(text_embeddings(ref_tokens + target_tokens)) + eos
        let combinedTextIds = MLX.concatenated([refTextIds, textIds], axis: 1)
        let combinedTextEmbedRaw = talker.embedText(combinedTextIds)
        var textEmbed = talker.textProjection(combinedTextEmbedRaw)
        textEmbed = MLX.concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLens = textEmbed.dim(1)

        // 5. Build codec_embed: codec_bos + sum_of_all_codebook_embeddings(ref_codes)
        // ref_codes shape: [1, 16, ref_time]
        guard let codePredictor = talker.codePredictor else {
            throw Qwen3TTSError.modelNotInitialized("Code predictor not available")
        }

        let firstCbCodes = refCodes[0..., 0, 0...]  // [1, ref_time]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCbCodes)
        let numCodeGroups = talkerConfig.numCodeGroups
        for i in 0..<(numCodeGroups - 1) {
            let cbCodes = refCodes[0..., i + 1, 0...]
            refCodecEmbed = refCodecEmbed + codePredictor.codecEmbedding[i](cbCodes)
        }

        // Prepend codec_bos
        let codecBosId = MLXArray([Int32(talkerConfig.codecBosId)]).expandedDimensions(axis: 0)
        let codecBosEmbed = talker.getInputEmbeddings()(codecBosId)
        let codecEmbedIcl = MLX.concatenated([codecBosEmbed, refCodecEmbed], axis: 1)  // [1, ref_time+1, hidden]
        let codecLens = codecEmbedIcl.dim(1)

        // 6. Non-streaming mode overlay (matching official Qwen3-TTS non_streaming_mode=True)
        // All text first (overlaid with codec_pad), then all codec (overlaid with tts_pad)
        let codecPadId = MLXArray([Int32(talkerConfig.codecPadId)]).expandedDimensions(axis: 0)
        let codecPadEmbed = talker.getInputEmbeddings()(codecPadId)

        // text + codec_pad
        let codecPadBroadcast = MLX.broadcast(codecPadEmbed, to: [1, textLens, codecPadEmbed.dim(2)])
        let textWithCodecPad = textEmbed + codecPadBroadcast

        // codec + tts_pad
        let ttsPadBroadcast = MLX.broadcast(ttsPadEmbed, to: [1, codecLens, ttsPadEmbed.dim(2)])
        let codecWithTextPad = codecEmbedIcl + ttsPadBroadcast

        let iclInputEmbed = MLX.concatenated([textWithCodecPad, codecWithTextPad], axis: 1)

        // 7. Language ID
        var languageId: Int? = nil
        let langLower = language.lowercased()
        if langLower != "auto" {
            languageId = talkerConfig.codecLanguageId[langLower]
        }

        // 8. Speaker embedding (ICL still uses x-vector)
        var speakerEmbed: MLXArray? = nil
        if speakerEncoder != nil {
            speakerEmbed = try extractSpeakerEmbedding(refAudio.ndim == 1 ? refAudio : refAudio.flattened())
        }

        // 9. Build codec prefix (think/nothink + speaker + pad + bos)
        var codecPrefill: [Int32]
        if languageId == nil {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(languageId!),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        let codecPrefillArray = MLXArray(codecPrefill).expandedDimensions(axis: 0)
        var codecPrefixEmbed = talker.getInputEmbeddings()(codecPrefillArray)

        let codecPrefixSuffixArray = MLXArray([
            Int32(talkerConfig.codecPadId),
            Int32(talkerConfig.codecBosId)
        ]).expandedDimensions(axis: 0)
        let codecPrefixSuffix = talker.getInputEmbeddings()(codecPrefixSuffixArray)

        if let speakerEmbed = speakerEmbed {
            codecPrefixEmbed = MLX.concatenated([
                codecPrefixEmbed,
                speakerEmbed.reshaped([1, 1, -1]),
                codecPrefixSuffix
            ], axis: 1)
        } else {
            codecPrefixEmbed = MLX.concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        // 10. Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleIds = targetIds[0..., 0..<3]
        let roleEmbedRaw = talker.embedText(roleIds)
        let roleEmbed = talker.textProjection(roleEmbedRaw)

        // 11. Build pad/bos prefix (text side overlaid with codec prefix[:-1])
        let prefixLen = codecPrefixEmbed.dim(1)
        let padCount = prefixLen - 2
        let padEmbeds = MLX.broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(2)])
        var combinedPrefix = MLX.concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0..<(prefixLen - 1), 0...]

        // 12. Full input_embeds: role + codec_prefix + icl_embed
        let inputEmbeds = MLX.concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)

        // For ICL mode, trailing text is just tts_pad (all text is in the prefill)
        let trailingTextHidden = ttsPadEmbed

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes)
    }

    // MARK: - Generation

    /// Generate audio using VoiceDesign mode
    public func generateVoiceDesign(
        text: String,
        language: String = "auto",
        instruct: String? = nil,
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        maxTokens: Int = 2048
    ) throws -> MLXArray {
        guard let talkerConfig = config.talkerConfig else {
            throw Qwen3TTSError.modelNotInitialized("Talker config not available")
        }

        guard speechTokenizer != nil else {
            throw Qwen3TTSError.modelNotInitialized("Speech tokenizer not loaded")
        }

        // Prepare inputs (no speaker for VoiceDesign mode)
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
            text: text,
            language: language,
            speaker: nil,
            instruct: instruct
        )

        // Cap max_tokens based on target text length to prevent runaway generation
        let targetTokenCount = tokenizer?.encode(text: text).count ?? text.count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Get EOS token ID
        let eosTokenId = talkerConfig.codecEosTokenId

        // Build suppress tokens list (special tokens except EOS)
        let vocabSize = talkerConfig.vocabSize
        var suppressTokens = [Int]()
        for i in (vocabSize - 1024)..<vocabSize {
            if i != eosTokenId {
                suppressTokens.append(i)
            }
        }

        // Initialize cache
        let cache = talker.makeCache()
        var generatedCodes: [[MLXArray]] = []
        var generatedTokens: [Int] = []

        // Current input
        var currentInput = inputEmbeds
        var trailingIdx = 0

        // Autoregressive generation
        for _ in 0..<effectiveMaxTokens {
            // Forward through Talker
            let (logits, hiddenStates) = talker(currentInput, cache: cache)
            eval(logits, hiddenStates)

            // Sample first codebook token
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokens,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId
            )

            let tokenValue = nextToken.item(Int.self)
            generatedTokens.append(tokenValue)

            // Check EOS
            if tokenValue == eosTokenId {
                break
            }

            // Generate remaining 15 codebooks using Code Predictor
            var codeTokens: [MLXArray] = [nextToken]

            if let codePredictor = talker.codePredictor {
                let seqLen = hiddenStates.dim(1)
                let codeHidden = hiddenStates[0..., (seqLen - 1)..., 0...]
                let codePredictorCache = codePredictor.makeCache()

                for codeIdx in 0..<15 {
                    // Prepare input
                    let codeInput: MLXArray
                    if codeIdx == 0 {
                        // Prefill: [hidden_state, code_0_embed]
                        let code0Embed = talker.getInputEmbeddings()(nextToken)
                        codeInput = MLX.concatenated([codeHidden, code0Embed], axis: 1)
                    } else {
                        // Generation: only previous code embedding
                        let prevCode = codeTokens[codeIdx]
                        let codeEmbed = codePredictor.codecEmbedding[codeIdx - 1](prevCode)
                        codeInput = codeEmbed
                    }

                    // Forward through Code Predictor
                    let (codeLogits, _, _) = codePredictor(
                        codeInput,
                        cache: codePredictorCache,
                        generationStep: codeIdx
                    )
                    eval(codeLogits)

                    // Sample
                    let nextCode = sampleToken(
                        codeLogits,
                        temperature: temperature,
                        topK: topK,
                        topP: topP
                    )
                    codeTokens.append(nextCode)
                }
            }

            // Store codebook tokens for later stacking
            generatedCodes.append(codeTokens)

            // Prepare next input
            // Get trailing text embedding
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Compute codec embedding (sum of all 16 codebook embeddings)
            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            if let codePredictor = talker.codePredictor {
                for (i, code) in codeTokens.dropFirst().enumerated() {
                    codecEmbed = codecEmbed + codePredictor.codecEmbedding[i](code)
                }
            }

            currentInput = textEmbed + codecEmbed
        }

        // Stack all generated codes: [1, seq_len, 16]
        guard !generatedCodes.isEmpty else {
            throw Qwen3TTSError.generationFailed("No tokens generated")
        }

        var codesArray: [MLXArray] = []
        for codeStep in generatedCodes {
            let stepCodes = MLX.concatenated(codeStep, axis: 1)  // [1, 16]
            codesArray.append(stepCodes)
        }
        let codes = MLX.stacked(codesArray, axis: 1)  // [1, seq_len, 16]

        // Decode to audio
        let (audio, audioLengths) = speechTokenizer!.decode(codes)

        // Trim to valid length
        let validLen = audioLengths[0].item(Int.self)
        var audioTrimmed = audio[0]  // Remove batch dimension

        if validLen > 0 && validLen < audioTrimmed.dim(0) {
            audioTrimmed = audioTrimmed[0..<validLen]
        }

        return audioTrimmed
    }

    /// Generate audio using CustomVoice mode with predefined speaker and optional emotion/style instruction
    ///
    /// This method is for CustomVoice model variants (e.g., Qwen3-TTS-12Hz-*-CustomVoice).
    /// It uses predefined speaker voices with optional emotion/style instructions.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - speaker: Speaker name (e.g., "Vivian", "Ryan", "Serena")
    ///   - language: Language code ("auto", "chinese", "english", etc.)
    ///   - instruct: Optional emotion/style instruction (e.g., "Very happy and excited.", "用愤怒的语气说")
    ///   - temperature: Sampling temperature (default: 0.9)
    ///   - topK: Top-k sampling (default: 50)
    ///   - topP: Top-p sampling (default: 1.0)
    ///   - repetitionPenalty: Repetition penalty (default: 1.05)
    ///   - maxTokens: Maximum tokens to generate (default: 2048)
    /// - Returns: Generated audio as MLXArray
    ///
    /// Example:
    /// ```swift
    /// let audio = try model.generateCustomVoice(
    ///     text: "Hello, how are you?",
    ///     speaker: "Vivian",
    ///     language: "english",
    ///     instruct: "Very happy and excited."
    /// )
    /// ```
    public func generateCustomVoice(
        text: String,
        speaker: String,
        language: String = "auto",
        instruct: String? = nil,
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        maxTokens: Int = 2048
    ) throws -> MLXArray {
        guard let talkerConfig = config.talkerConfig else {
            throw Qwen3TTSError.modelNotInitialized("Talker config not available")
        }

        guard speechTokenizer != nil else {
            throw Qwen3TTSError.modelNotInitialized("Speech tokenizer not loaded")
        }

        // Validate speaker
        if let spkIdMap = talkerConfig.spkId {
            let validSpeakers = Array(spkIdMap.keys)
            if !validSpeakers.contains(speaker.lowercased()) {
                throw Qwen3TTSError.invalidInput("Speaker '\(speaker)' not found. Available speakers: \(validSpeakers.joined(separator: ", "))")
            }
        } else {
            throw Qwen3TTSError.invalidInput("This model does not support CustomVoice. No speakers defined.")
        }

        // Prepare inputs with speaker
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct
        )

        // Cap max_tokens based on target text length to prevent runaway generation
        let targetTokenCount = tokenizer?.encode(text: text).count ?? text.count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Get EOS token ID
        let eosTokenId = talkerConfig.codecEosTokenId

        // Build suppress tokens list (special tokens except EOS)
        let vocabSize = talkerConfig.vocabSize
        var suppressTokens = [Int]()
        for i in (vocabSize - 1024)..<vocabSize {
            if i != eosTokenId {
                suppressTokens.append(i)
            }
        }

        // Initialize cache
        let cache = talker.makeCache()
        var generatedCodes: [[MLXArray]] = []
        var generatedTokens: [Int] = []

        // Current input
        var currentInput = inputEmbeds
        var trailingIdx = 0

        // Autoregressive generation
        for _ in 0..<effectiveMaxTokens {
            // Forward through Talker
            let (logits, hiddenStates) = talker(currentInput, cache: cache)
            eval(logits, hiddenStates)

            // Sample first codebook token
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokens,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId
            )

            let tokenValue = nextToken.item(Int.self)
            generatedTokens.append(tokenValue)

            // Check EOS
            if tokenValue == eosTokenId {
                break
            }

            // Generate remaining 15 codebooks using Code Predictor
            var codeTokens: [MLXArray] = [nextToken]

            if let codePredictor = talker.codePredictor {
                let seqLen = hiddenStates.dim(1)
                let codeHidden = hiddenStates[0..., (seqLen - 1)..., 0...]
                let codePredictorCache = codePredictor.makeCache()

                for codeIdx in 0..<15 {
                    // Prepare input
                    let codeInput: MLXArray
                    if codeIdx == 0 {
                        // Prefill: [hidden_state, code_0_embed]
                        let code0Embed = talker.getInputEmbeddings()(nextToken)
                        codeInput = MLX.concatenated([codeHidden, code0Embed], axis: 1)
                    } else {
                        // Generation: only previous code embedding
                        let prevCode = codeTokens[codeIdx]
                        let codeEmbed = codePredictor.codecEmbedding[codeIdx - 1](prevCode)
                        codeInput = codeEmbed
                    }

                    // Forward through Code Predictor
                    let (codeLogits, _, _) = codePredictor(
                        codeInput,
                        cache: codePredictorCache,
                        generationStep: codeIdx
                    )
                    eval(codeLogits)

                    // Sample
                    let nextCode = sampleToken(
                        codeLogits,
                        temperature: temperature,
                        topK: topK,
                        topP: topP
                    )
                    codeTokens.append(nextCode)
                }
            }

            // Store codebook tokens for later stacking
            generatedCodes.append(codeTokens)

            // Prepare next input
            // Get trailing text embedding
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Compute codec embedding (sum of all 16 codebook embeddings)
            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            if let codePredictor = talker.codePredictor {
                for (i, code) in codeTokens.dropFirst().enumerated() {
                    codecEmbed = codecEmbed + codePredictor.codecEmbedding[i](code)
                }
            }

            currentInput = textEmbed + codecEmbed
        }

        // Stack all generated codes: [1, seq_len, 16]
        guard !generatedCodes.isEmpty else {
            throw Qwen3TTSError.generationFailed("No tokens generated")
        }

        var codesArray: [MLXArray] = []
        for codeStep in generatedCodes {
            let stepCodes = MLX.concatenated(codeStep, axis: 1)  // [1, 16]
            codesArray.append(stepCodes)
        }
        let codes = MLX.stacked(codesArray, axis: 1)  // [1, seq_len, 16]

        // Decode to audio
        let (audio, audioLengths) = speechTokenizer!.decode(codes)

        // Trim to valid length
        let validLen = audioLengths[0].item(Int.self)
        var audioTrimmed = audio[0]  // Remove batch dimension

        if validLen > 0 && validLen < audioTrimmed.dim(0) {
            audioTrimmed = audioTrimmed[0..<validLen]
        }

        return audioTrimmed
    }

    /// Get list of supported speakers for CustomVoice mode
    public var supportedSpeakers: [String] {
        guard let talkerConfig = config.talkerConfig,
              let spkIdMap = talkerConfig.spkId else {
            return []
        }
        return Array(spkIdMap.keys).sorted()
    }

    // MARK: - Voice Cloning

    /// Generate audio using Voice Cloning (ICL mode)
    ///
    /// Voice cloning uses In-Context Learning (ICL) to clone a voice from reference audio.
    /// The reference audio and its transcript are used as context for generating speech
    /// in the same voice.
    ///
    /// **Note**: This feature requires the speech tokenizer encoder, which is not yet
    /// implemented in Swift. Currently, only speaker embedding extraction is available.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize in the cloned voice
    ///   - referenceAudio: Reference audio waveform [samples] at 24kHz
    ///   - referenceText: Transcript of the reference audio
    ///   - language: Language code ("auto", "chinese", "english", etc.)
    ///   - temperature: Sampling temperature (default: 0.9)
    ///   - topK: Top-k sampling (default: 50)
    ///   - topP: Top-p sampling (default: 1.0)
    ///   - repetitionPenalty: Repetition penalty (default: 1.5 for ICL mode)
    ///   - maxTokens: Maximum tokens to generate (default: 2048)
    /// - Returns: Generated audio as MLXArray
    ///
    /// Example:
    /// ```swift
    /// // Load reference audio (must be 24kHz)
    /// let refAudio = try loadAudio("reference.wav", sampleRate: 24000)
    /// let refText = "This is what I said in the reference audio."
    ///
    /// let audio = try await model.generateVoiceClone(
    ///     text: "Hello, this is my cloned voice!",
    ///     referenceAudio: refAudio,
    ///     referenceText: refText,
    ///     language: "english"
    /// )
    /// ```
    public func generateVoiceClone(
        text: String,
        referenceAudio: MLXArray,
        referenceText: String,
        language: String = "auto",
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.5,
        maxTokens: Int = 2048
    ) throws -> MLXArray {
        // Voice cloning requires:
        // 1. Speech tokenizer encoder to encode reference audio to codes
        // 2. Speaker encoder for x-vector embedding

        guard let talkerConfig = config.talkerConfig else {
            throw Qwen3TTSError.modelNotInitialized("Talker config not available")
        }

        guard let speechTokenizer = speechTokenizer else {
            throw Qwen3TTSError.modelNotInitialized("Speech tokenizer not loaded")
        }

        guard speechTokenizer.hasEncoder else {
            throw Qwen3TTSError.modelNotInitialized(
                "Voice cloning (ICL mode) requires the speech tokenizer encoder. " +
                "Make sure to load a model with encoder weights."
            )
        }

        // 1. Prepare ICL inputs
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes) = try prepareICLGenerationInputs(
            text: text,
            refAudio: referenceAudio,
            refText: referenceText,
            language: language
        )

        // Cap max_tokens based on target text length to prevent runaway generation
        // At 12.5 Hz codec rate, ~3-5 codec tokens per text token is typical speech
        // Factor of 6 gives ~50% margin for slow speech / pauses
        let targetTokenCount = tokenizer?.encode(text: text).count ?? text.count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // 2. Initialize cache and generation state
        let cache = talker.makeCache()
        var generatedCodes: [[MLXArray]] = []
        var generatedTokens: [Int] = []
        let eosTokenId = talkerConfig.codecEosTokenId

        // Build suppress tokens list (special tokens except EOS)
        let vocabSize = talkerConfig.vocabSize
        var suppressTokens = [Int]()
        for i in (vocabSize - 1024)..<vocabSize {
            if i != eosTokenId {
                suppressTokens.append(i)
            }
        }

        var trailingIdx = 0
        var currentInput = inputEmbeds

        // Get code predictor (required for generation)
        guard let codePredictor = talker.codePredictor else {
            throw Qwen3TTSError.modelNotInitialized("Code predictor not available")
        }

        // 3. Autoregressive generation
        for _ in 0..<effectiveMaxTokens {
            // Forward through Talker
            let (logits, hiddenStates) = talker(currentInput, cache: cache)
            eval(logits, hiddenStates)

            // Sample first codebook token
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokens,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId
            )

            let tokenValue = Int(nextToken[0, 0].item(Int32.self))

            // Check for EOS
            if tokenValue == eosTokenId {
                break
            }

            generatedTokens.append(tokenValue)

            // Generate remaining codebook tokens with code predictor
            var codeTokens = [nextToken]
            let codeHidden = hiddenStates[0..., (-1)..., 0...]

            let codeCache = codePredictor.makeCache()

            for codeIdx in 0..<(talkerConfig.numCodeGroups - 1) {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    // Prefill: concatenate [hidden_state, code_0_embed]
                    let code0Embed = talker.getInputEmbeddings()(nextToken)
                    codeInput = MLX.concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    // Generation: just pass embedding of previous code token
                    let codeEmbed = codePredictor.codecEmbedding[codeIdx - 1](codeTokens[codeIdx])
                    codeInput = codeEmbed
                }

                // Code predictor forward
                let (codeLogits, _, _) = codePredictor(
                    codeInput,
                    cache: codeCache,
                    generationStep: codeIdx
                )

                // Sample
                let nextCode = sampleToken(
                    codeLogits,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: 1.0,
                    generatedTokens: [],
                    suppressTokens: nil,
                    eosTokenId: nil
                )
                codeTokens.append(nextCode)
            }

            // Stack all codebook tokens
            generatedCodes.append(codeTokens)

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Build codec embedding for next step (sum of all codebook embeddings)
            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            for (i, code) in codeTokens.dropFirst().enumerated() {
                codecEmbed = codecEmbed + codePredictor.codecEmbedding[i](code)
            }

            currentInput = textEmbed + codecEmbed
            eval(currentInput)
        }

        guard !generatedCodes.isEmpty else {
            throw Qwen3TTSError.generationFailed("No tokens generated")
        }

        // 4. Stack generated codes: [batch, seq_len, num_code_groups]
        let genCodesStacked = MLX.stacked(
            generatedCodes.map { codes in MLX.concatenated(codes, axis: 1) },
            axis: 1
        )

        // 5. Prepend reference codes for decoding
        // ref_codes: [1, 16, ref_time] -> [1, ref_time, 16]
        let refCodesT = refCodes.transposed(0, 2, 1)
        // Combine: [1, ref_time + gen_len, 16]
        let fullCodes = MLX.concatenated([refCodesT, genCodesStacked], axis: 1)

        let refLen = refCodes.dim(2)
        let totalLen = fullCodes.dim(1)

        // 6. Decode full codes to audio
        let (audio, audioLengths) = speechTokenizer.decode(fullCodes)
        var audioOutput = audio[0]  // Remove batch dim

        // Trim to valid length
        let validLen = Int(audioLengths[0].item(Int32.self))
        if validLen > 0 && validLen < audioOutput.dim(0) {
            audioOutput = audioOutput[0..<validLen]
        }

        // 7. Remove the reference audio portion using proportional trimming
        let cut = Int(Float(refLen) / Float(max(totalLen, 1)) * Float(audioOutput.dim(0)))
        if cut > 0 && cut < audioOutput.dim(0) {
            audioOutput = audioOutput[cut...]
        }

        eval(audioOutput)
        return audioOutput
    }

    /// Check if voice cloning is supported
    ///
    /// Voice cloning requires:
    /// - Base model type (not VoiceDesign or CustomVoice)
    /// - Speech tokenizer with encoder capability
    ///
    /// Currently returns false as the speech tokenizer encoder is not yet implemented.
    public var supportsVoiceCloning: Bool {
        guard config.ttsModelType == "base" else { return false }
        guard let tokenizer = speechTokenizer else { return false }
        return tokenizer.hasEncoder
    }

    // MARK: - Weight Loading

    /// Sanitize weights from PyTorch format
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip position_ids
            if key.contains("position_ids") {
                continue
            }

            var newValue = value

            // Conv1d weight conversion: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            let isConvWeight = (key.contains("conv") || key.contains("speaker_encoder.fc"))
                && key.contains("weight")
            if isConvWeight && value.ndim == 3 {
                if !checkArrayShapeQwen3(value) {
                    newValue = value.transposed(0, 2, 1)
                }
            }

            sanitized[key] = newValue
        }

        return sanitized
    }

    /// Check if Conv1d weight is already in MLX format
    private static func checkArrayShapeQwen3(_ arr: MLXArray) -> Bool {
        let shape = arr.shape
        guard shape.count == 3 else { return false }

        let (_, dim2, dim3) = (shape[0], shape[1], shape[2])

        // Heuristic: kernel_size is usually small
        if dim2 == 1 {
            return dim3 > 64  // dim3 large -> MLX format
        } else if dim3 == 1 {
            return dim2 <= 64  // dim2 small -> MLX format
        }

        return dim2 < dim3
    }

    public var sampleRate: Int {
        return config.sampleRate
    }

    // MARK: - Public API

    /// The model type (voice_design, custom_voice, or base)
    public var ttsModelType: String {
        return config.ttsModelType
    }

    /// Generate audio from text
    ///
    /// Automatically routes to the appropriate generation method based on model type:
    /// - voice_design: Uses `generateVoiceDesign()` with instruct as voice description
    /// - custom_voice: Uses `generateCustomVoice()` with speaker and optional instruct
    /// - base: Uses `generateCustomVoice()` with speaker only (no instruct)
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - speaker: Speaker name for CustomVoice/Base models (e.g., "Vivian", "Ryan")
    ///   - instruct: Voice description (VoiceDesign) or emotion/style (CustomVoice)
    ///   - language: Language code ("auto", "chinese", "english", etc.)
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling
    ///   - topP: Top-p sampling
    ///   - repetitionPenalty: Repetition penalty
    ///   - maxTokens: Maximum tokens to generate
    /// - Returns: Generated audio as MLXArray
    public func generate(
        text: String,
        speaker: String? = nil,
        instruct: String? = nil,
        language: String = "auto",
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        maxTokens: Int = 2048
    ) async throws -> MLXArray {
        switch config.ttsModelType {
        case "voice_design":
            guard instruct != nil else {
                throw Qwen3TTSError.invalidInput(
                    "VoiceDesign model requires 'instruct' to describe the voice " +
                    "(e.g., 'A cheerful young female voice with high pitch')"
                )
            }
            return try generateVoiceDesign(
                text: text,
                language: language,
                instruct: instruct,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens
            )

        case "custom_voice":
            guard let speaker = speaker else {
                throw Qwen3TTSError.invalidInput(
                    "CustomVoice model requires 'speaker' (e.g., 'Vivian', 'Ryan'). " +
                    "Available speakers: \(supportedSpeakers.joined(separator: ", "))"
                )
            }
            return try generateCustomVoice(
                text: text,
                speaker: speaker,
                language: language,
                instruct: instruct,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens
            )

        case "base":
            guard let speaker = speaker else {
                throw Qwen3TTSError.invalidInput(
                    "Base model requires 'speaker' (e.g., 'Vivian', 'Ryan'). " +
                    "Available speakers: \(supportedSpeakers.joined(separator: ", "))"
                )
            }
            // Base model uses CustomVoice generation without instruct
            return try generateCustomVoice(
                text: text,
                speaker: speaker,
                language: language,
                instruct: nil,  // Base model doesn't support instruct
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens
            )

        default:
            // Default to VoiceDesign for unknown types
            return try generateVoiceDesign(
                text: text,
                language: language,
                instruct: instruct,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens
            )
        }
    }

    /// Generate with streaming output
    ///
    /// See `generate()` for parameter documentation and model type routing.
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
            Task {
                do {
                    let startTime = Date()

                    let audio = try await self.generate(
                        text: text,
                        speaker: speaker,
                        instruct: instruct,
                        language: language,
                        temperature: temperature,
                        topK: topK,
                        topP: topP,
                        repetitionPenalty: repetitionPenalty,
                        maxTokens: maxTokens
                    )

                    let totalTime = Date().timeIntervalSince(startTime)

                    let info = Qwen3TTSGenerationInfo(
                        promptTokenCount: 0,
                        generationTokenCount: 0,
                        prefillTime: 0,
                        generateTime: totalTime,
                        tokensPerSecond: 0,
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

    // MARK: - Model Loading

    /// Load model from a local directory or HuggingFace Hub repository.
    ///
    /// - Parameter modelPath: Either a local directory path (e.g. "/path/to/model")
    ///   or a HuggingFace repo ID (e.g. "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    /// - Returns: The loaded model ready for generation
    public static func fromPretrained(_ modelPath: String) async throws -> Qwen3TTSModel {
        let modelDir: URL

        if isHuggingFaceRepoID(modelPath) {
            modelDir = try await resolveOrDownloadModel(repoID: modelPath)
        } else {
            modelDir = URL(fileURLWithPath: modelPath)
        }

        // Load config
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)

        // Load weights first (need to detect quantized embeddings before model creation)
        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        // Collect which embeddings are quantized (have .scales keys in weights)
        let quantizedEmbeddingPaths: Set<String> = Set(
            weights.keys
                .filter { $0.contains("embedding") && $0.hasSuffix(".scales") }
                .map { String($0.dropLast(".scales".count)) }
        )

        // Create model
        let model = Qwen3TTSModel(config)

        // Quantize model layers to match weight format
        if let quantization = config.quantization {
            quantize(
                model: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                mode: quantization.mode,
                filter: { path, module in
                    if module is Embedding {
                        // Only quantize this specific embedding if its weights are quantized
                        return quantizedEmbeddingPaths.contains(path)
                    }
                    return true
                }
            )
            let embedStr = quantizedEmbeddingPaths.isEmpty ? "" : "+embeddings(\(quantizedEmbeddingPaths.sorted().joined(separator: ",")))"
            debugPrint("🔊 Applied quantization: \(quantization.bits)-bit, group_size=\(quantization.groupSize)\(embedStr)")
        }

        // Sanitize weights
        var sanitizedWeights = sanitize(weights: weights)

        // Extract text_token_map before update() (it's not a Module parameter)
        let tokenMapKey = "talker.model.text_token_map"
        let textTokenMap = sanitizedWeights.removeValue(forKey: tokenMapKey)

        // Apply weights (allow unused keys for models with optional components)
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [])

        // Assign token map after weight loading
        if let tokenMap = textTokenMap {
            model.talker.model.textTokenMap = tokenMap
            debugPrint("🔊 Loaded pruned vocabulary token map: \(tokenMap.shape)")
        }

        eval(model)

        // Post-load hook
        try await model.postLoadHook(modelDir: modelDir)

        return model
    }

    // MARK: - HuggingFace Hub Support

    /// Check if the given string looks like a HuggingFace repo ID (e.g. "org/model-name")
    /// rather than a local file path.
    private static func isHuggingFaceRepoID(_ path: String) -> Bool {
        // Local paths start with /, ~, or .
        if path.hasPrefix("/") || path.hasPrefix("~") || path.hasPrefix(".") {
            return false
        }
        // A HuggingFace repo ID has exactly one slash: "org/model"
        let components = path.split(separator: "/")
        return components.count == 2
    }

    /// Resolve a HuggingFace repo to a local directory, downloading if needed.
    private static func resolveOrDownloadModel(repoID: String) async throws -> URL {
        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        guard let repo = Repo.ID(rawValue: repoID) else {
            throw AudioGenerationError.invalidInput("Invalid HuggingFace repository ID: \(repoID)")
        }

        // Cache directory: ~/Library/Caches/qwen3-tts/org_model
        let modelSubdir = repoID.replacingOccurrences(of: "/", with: "_")
        let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-tts")
            .appendingPathComponent(modelSubdir)

        // Check if model already cached (config.json + safetensors exist)
        let configPath = modelDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configPath.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let hasSafetensors = files?.contains { $0.pathExtension == "safetensors" } ?? false

            if hasSafetensors {
                print("Using cached model at: \(modelDir.path)")
                return modelDir
            }
        }

        // Download from HuggingFace Hub
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Clean up any partial downloads
        if let existingFiles = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil) {
            for file in existingFiles {
                try? FileManager.default.removeItem(at: file)
            }
        }

        print("Downloading model \(repoID)...")
        _ = try await client.downloadSnapshot(
            of: repo,
            kind: .model,
            to: modelDir,
            revision: "main",
            progressHandler: { progress in
                print("  \(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }

    /// Post-load initialization
    public func postLoadHook(modelDir: URL) async throws {
        // Load tokenizer
        if tokenizer == nil {
            tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        }

        // Load speech tokenizer
        let speechTokenizerPath = modelDir.appendingPathComponent("speech_tokenizer")
        if FileManager.default.fileExists(atPath: speechTokenizerPath.path) {
            // Load config
            let configPath = speechTokenizerPath.appendingPathComponent("config.json")
            let configData = try Data(contentsOf: configPath)
            let tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: configData)

            // Create tokenizer
            let tokenizer = Qwen3TTSSpeechTokenizer(tokenizerConfig)

            // Load weights
            var tokenizerWeights: [String: MLXArray] = [:]
            let files = try FileManager.default.contentsOfDirectory(at: speechTokenizerPath, includingPropertiesForKeys: nil)
            let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

            for file in safetensorFiles {
                let fileWeights = try MLX.loadArrays(url: file)
                tokenizerWeights.merge(fileWeights) { _, new in new }
            }

            // Sanitize and load (skip strict verification for now)
            let sanitizedWeights = sanitizeSpeechTokenizerWeights(tokenizerWeights)

            // Load weights
            let unflattened = ModuleParameters.unflattened(sanitizedWeights)
            try tokenizer.update(parameters: unflattened, verify: [])
            eval(tokenizer)

            // Initialize encoder codebooks (compute embeddings from raw data)
            tokenizer.initializeEncoderCodebooks()

            self.speechTokenizer = tokenizer
        }
    }

    /// Sanitize speech tokenizer weights
    private func sanitizeSpeechTokenizerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var decoderCodebookData: [String: [String: MLXArray]] = [:]
        var encoderCodebookData: [String: [String: MLXArray]] = [:]

        // Mapping for decoder.decoder array indices to named keys
        let decoderIndexMapping: [String: String] = [
            "decoder.decoder.0": "decoder.decoder.initConv",
            "decoder.decoder.1": "decoder.decoder.block0",
            "decoder.decoder.2": "decoder.decoder.block1",
            "decoder.decoder.3": "decoder.decoder.block2",
            "decoder.decoder.4": "decoder.decoder.block3",
            "decoder.decoder.5": "decoder.decoder.outSnake",
            "decoder.decoder.6": "decoder.decoder.outConv",
        ]

        // Mapping for encoder.encoder.layers to Swift structure
        // Python: encoder.encoder.layers.X -> Swift: encoder.{init_conv1d, layers.Y, final_conv1d}
        // Layer mapping: 0=init, 1=res0, 3=down0, 4=res1, 6=down1, 7=res2, 9=down2, 10=res3, 12=down3, 14=final
        let encoderSeanetMapping: [String: String] = [
            "encoder.encoder.layers.0.": "encoder.encoder.init_conv1d.",
            "encoder.encoder.layers.1.": "encoder.encoder.layers.0.residuals.0.",
            "encoder.encoder.layers.3.": "encoder.encoder.layers.0.downsample.",
            "encoder.encoder.layers.4.": "encoder.encoder.layers.1.residuals.0.",
            "encoder.encoder.layers.6.": "encoder.encoder.layers.1.downsample.",
            "encoder.encoder.layers.7.": "encoder.encoder.layers.2.residuals.0.",
            "encoder.encoder.layers.9.": "encoder.encoder.layers.2.downsample.",
            "encoder.encoder.layers.10.": "encoder.encoder.layers.3.residuals.0.",
            "encoder.encoder.layers.12.": "encoder.encoder.layers.3.downsample.",
            "encoder.encoder.layers.14.": "encoder.encoder.final_conv1d.",
        ]

        for (key, value) in weights {
            // Handle decoder codebook data (uses ._codebook. format)
            if key.contains("._codebook.cluster_usage") || key.contains("._codebook.embedding_sum") {
                let basePath = key.components(separatedBy: "._codebook.").first!
                if decoderCodebookData[basePath] == nil {
                    decoderCodebookData[basePath] = [:]
                }
                if key.contains("cluster_usage") {
                    decoderCodebookData[basePath]!["cluster_usage"] = value
                } else {
                    decoderCodebookData[basePath]!["embedding_sum"] = value
                }
                continue
            }

            // Handle encoder codebook data (uses .codebook. format)
            if key.hasPrefix("encoder.quantizer.") && key.contains(".codebook.") {
                // encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum
                // encoder.quantizer.acoustic_residual_vector_quantizer.layers.X.codebook.embed_sum
                let components = key.components(separatedBy: ".codebook.")
                if components.count == 2 {
                    let basePath = components[0]
                    let fieldName = components[1]
                    if fieldName == "embed_sum" || fieldName == "cluster_usage" {
                        if encoderCodebookData[basePath] == nil {
                            encoderCodebookData[basePath] = [:]
                        }
                        encoderCodebookData[basePath]![fieldName] = value
                        continue
                    }
                }
                // Skip initialized flag
                if key.contains(".initialized") {
                    continue
                }
            }

            var newKey = key
            var newValue = value

            // === DECODER KEY REMAPPING ===

            // Remap decoder.decoder array indices to named keys
            for (indexPrefix, namedPrefix) in decoderIndexMapping {
                if key.hasPrefix(indexPrefix) {
                    newKey = key.replacingOccurrences(of: indexPrefix, with: namedPrefix)
                    break
                }
            }

            // Remap DecoderBlock internal keys: block.X -> named keys
            if newKey.hasPrefix("decoder.") {
                newKey = newKey
                    .replacingOccurrences(of: ".block.0.", with: ".snake.")
                    .replacingOccurrences(of: ".block.1.", with: ".upsample.")
                    .replacingOccurrences(of: ".block.2.", with: ".res1.")
                    .replacingOccurrences(of: ".block.3.", with: ".res2.")
                    .replacingOccurrences(of: ".block.4.", with: ".res3.")
            }

            // === ENCODER KEY REMAPPING ===

            if newKey.hasPrefix("encoder.") {
                // Remap Seanet encoder layers
                for (pythonPrefix, swiftPrefix) in encoderSeanetMapping {
                    if newKey.hasPrefix(pythonPrefix) {
                        newKey = newKey.replacingOccurrences(of: pythonPrefix, with: swiftPrefix)
                        break
                    }
                }

                // Remap encoder residual block indices
                // Python: block.1, block.3 -> Swift: block.0, block.1
                if newKey.contains(".residuals.") {
                    newKey = newKey
                        .replacingOccurrences(of: ".block.1.", with: ".block.0.")
                        .replacingOccurrences(of: ".block.3.", with: ".block.1.")
                }

                // Fix Seanet conv path: StreamableConv1d.conv -> NormConv1d.conv -> EncoderConv1d
                // Python: .conv.weight/bias -> Swift: .conv.conv.weight/bias
                // Only apply to encoder.encoder.* (Seanet), not encoder.encoder_transformer or encoder.quantizer
                let isSeanetConv = newKey.hasPrefix("encoder.encoder.") &&
                    !newKey.contains("encoder_transformer") &&
                    !newKey.contains("quantizer") &&
                    (newKey.contains(".conv.weight") || newKey.contains(".conv.bias"))
                if isSeanetConv {
                    newKey = newKey
                        .replacingOccurrences(of: ".conv.weight", with: ".conv.conv.weight")
                        .replacingOccurrences(of: ".conv.bias", with: ".conv.conv.bias")

                    // Force transpose conv weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
                    if newKey.hasSuffix(".weight") && value.ndim == 3 {
                        newValue = value.transposed(0, 2, 1)
                    }
                }

                // Remap encoder transformer keys
                // Python: encoder.encoder_transformer.layers.X -> Swift: encoder.encoder_transformer.transformer.layers.X
                if newKey.contains("encoder.encoder_transformer.layers.") {
                    newKey = newKey.replacingOccurrences(
                        of: "encoder.encoder_transformer.layers.",
                        with: "encoder.encoder_transformer.transformer.layers."
                    )

                    // LayerNorm naming
                    newKey = newKey
                        .replacingOccurrences(of: ".input_layernorm.", with: ".norm1.")
                        .replacingOccurrences(of: ".post_attention_layernorm.", with: ".norm2.")

                    // MLP naming
                    newKey = newKey
                        .replacingOccurrences(of: ".mlp.fc1.", with: ".gating.linear1.")
                        .replacingOccurrences(of: ".mlp.fc2.", with: ".gating.linear2.")

                    // Layer scale naming
                    newKey = newKey
                        .replacingOccurrences(of: ".self_attn_layer_scale.", with: ".layer_scale_1.")
                        .replacingOccurrences(of: ".mlp_layer_scale.", with: ".layer_scale_2.")
                }

                // Fix encoder.downsample.conv path (EncoderConvDownsample1d.conv -> StreamableConv1d.conv -> NormConv1d.conv)
                if newKey.hasPrefix("encoder.downsample.conv.") && !newKey.contains("encoder.downsample.conv.conv.") {
                    let isWeight = newKey.hasSuffix(".weight")
                    newKey = newKey.replacingOccurrences(of: "encoder.downsample.conv.", with: "encoder.downsample.conv.conv.conv.")
                    // Force transpose conv weights for encoder.downsample
                    if isWeight && value.ndim == 3 {
                        newValue = value.transposed(0, 2, 1)
                    }
                }

                // Remap encoder quantizer keys
                // Python: semantic_residual_vector_quantizer -> Swift: rvq_first
                // Python: acoustic_residual_vector_quantizer -> Swift: rvq_rest
                if newKey.contains("encoder.quantizer.") {
                    newKey = newKey
                        .replacingOccurrences(of: ".semantic_residual_vector_quantizer.", with: ".rvq_first.")
                        .replacingOccurrences(of: ".acoustic_residual_vector_quantizer.", with: ".rvq_rest.")
                    // layers.X -> vq.layers.X
                    newKey = newKey.replacingOccurrences(
                        of: RegexPattern(".rvq_(first|rest).layers."),
                        with: { match in
                            let rvqType = match.contains("first") ? "rvq_first" : "rvq_rest"
                            return ".\(rvqType).vq.layers."
                        }
                    )
                }
            }

            // === WEIGHT TRANSFORMATIONS ===

            // Check if this was a Seanet conv weight (already force-transposed above)
            let wasSeanetConvWeight = newKey.hasPrefix("encoder.encoder.") &&
                !newKey.contains("encoder_transformer") &&
                !newKey.contains("quantizer") &&
                newKey.hasSuffix(".conv.conv.weight")

            // input_proj/output_proj weight conversion for quantizer: [out, in, 1] -> [out, 1, in]
            let isProjWeight = (newKey.contains("input_proj.weight") || newKey.contains("output_proj.weight"))
                && newKey.contains("quantizer")
            if isProjWeight && value.ndim == 3 {
                newValue = value.transposed(0, 2, 1)
            }

            // Conv weight conversion: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            // Skip if already handled by Seanet conv processing above
            if newKey.contains("conv.weight") && value.ndim == 3 && !isProjWeight && !wasSeanetConvWeight {
                if !Qwen3TTSModel.checkArrayShapeQwen3(value) {
                    newValue = value.transposed(0, 2, 1)
                }
            }

            // ConvTranspose weight conversion (decoder only)
            // PyTorch ConvTranspose1d: [in, out, kernel] -> MLX ConvTransposed1d: [out, kernel, in]
            let isTransposeConv =
                (newKey.contains("upsample") && newKey.contains(".0.conv.weight"))
                || (newKey.contains("decoder.decoder.block") && newKey.contains("upsample.conv.weight"))
            if isTransposeConv && value.ndim == 3 {
                if !Qwen3TTSModel.checkArrayShapeQwen3(value) {
                    newValue = value.transposed(1, 2, 0)
                }
            }

            sanitized[newKey] = newValue
        }

        // Compute decoder embeddings from codebook data
        let eps: Float = 1e-5
        for (basePath, data) in decoderCodebookData {
            if let clusterUsage = data["cluster_usage"],
               let embeddingSum = data["embedding_sum"] {
                let embedding = embeddingSum / MLX.clip(clusterUsage.expandedDimensions(axis: 1), min: eps, max: Float.greatestFiniteMagnitude)
                sanitized["\(basePath).codebook.embed.weight"] = embedding
            }
        }

        // Compute encoder embeddings from codebook data
        for (basePath, data) in encoderCodebookData {
            if let clusterUsage = data["cluster_usage"],
               let embedSum = data["embed_sum"] {
                // Store raw data for encoder codebooks (they use embed_sum/cluster_usage directly)
                var newBasePath = basePath
                    .replacingOccurrences(of: ".semantic_residual_vector_quantizer.", with: ".rvq_first.")
                    .replacingOccurrences(of: ".acoustic_residual_vector_quantizer.", with: ".rvq_rest.")

                // Fix: layers.X -> vq.layers.X
                if let range = newBasePath.range(of: ".rvq_first.layers.") {
                    newBasePath = newBasePath.replacingCharacters(in: range, with: ".rvq_first.vq.layers.")
                } else if let range = newBasePath.range(of: ".rvq_rest.layers.") {
                    newBasePath = newBasePath.replacingCharacters(in: range, with: ".rvq_rest.vq.layers.")
                }

                let finalEmbedKey = "\(newBasePath).codebook.embeddingSum"
                let finalUsageKey = "\(newBasePath).codebook.clusterUsage"
                sanitized[finalEmbedKey] = embedSum
                sanitized[finalUsageKey] = clusterUsage
            }
        }

        return sanitized
    }

    /// Helper for regex-like replacement
    private func replacePattern(_ input: String, pattern: String, replacement: (String) -> String) -> String {
        // Simple replacement without regex for now
        return input
    }
}

// String extension for pattern replacement
private extension String {
    func replacingOccurrences(of pattern: RegexPattern, with replacement: (String) -> String) -> String {
        // For encoder quantizer: handle rvq_first.layers. and rvq_rest.layers. -> rvq_X.vq.layers.
        var result = self
        if result.contains(".rvq_first.layers.") {
            result = result.replacingOccurrences(of: ".rvq_first.layers.", with: ".rvq_first.vq.layers.")
        }
        if result.contains(".rvq_rest.layers.") {
            result = result.replacingOccurrences(of: ".rvq_rest.layers.", with: ".rvq_rest.vq.layers.")
        }
        return result
    }
}

private struct RegexPattern {
    let pattern: String
    init(_ pattern: String) {
        self.pattern = pattern
    }
}
