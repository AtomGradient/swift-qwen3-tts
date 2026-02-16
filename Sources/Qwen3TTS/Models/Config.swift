//
//  Config.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS Configuration
//  Ported from Python mlx-audio implementation
//

import Foundation
import MLXLMCommon

// MARK: - Dialect Value (can be false or a dialect name string)

/// Represents a dialect value that can be either false (not a dialect) or a dialect name string
public enum DialectValue: Codable, Sendable, Equatable {
    case notDialect
    case dialect(String)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Try to decode as Bool first (for false values)
        if let boolValue = try? container.decode(Bool.self) {
            if boolValue {
                throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected false or string")
            }
            self = .notDialect
        } else if let stringValue = try? container.decode(String.self) {
            self = .dialect(stringValue)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected false or string")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .notDialect:
            try container.encode(false)
        case .dialect(let name):
            try container.encode(name)
        }
    }

    /// Returns the dialect name if this is a dialect, nil otherwise
    public var dialectName: String? {
        switch self {
        case .notDialect: return nil
        case .dialect(let name): return name
        }
    }
}

// MARK: - Speaker Encoder Configuration

public struct Qwen3TTSSpeakerEncoderConfig: Codable, Sendable {
    public var melDim: Int
    public var encDim: Int
    public var encChannels: [Int]
    public var encKernelSizes: [Int]
    public var encDilations: [Int]
    public var encAttentionChannels: Int
    public var encRes2netScale: Int
    public var encSeChannels: Int
    public var sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case melDim = "mel_dim"
        case encDim = "enc_dim"
        case encChannels = "enc_channels"
        case encKernelSizes = "enc_kernel_sizes"
        case encDilations = "enc_dilations"
        case encAttentionChannels = "enc_attention_channels"
        case encRes2netScale = "enc_res2net_scale"
        case encSeChannels = "enc_se_channels"
        case sampleRate = "sample_rate"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.melDim = try container.decodeIfPresent(Int.self, forKey: .melDim) ?? 128
        self.encDim = try container.decodeIfPresent(Int.self, forKey: .encDim) ?? 1024
        self.encChannels = try container.decodeIfPresent([Int].self, forKey: .encChannels) ?? [512, 512, 512, 512, 1536]
        self.encKernelSizes = try container.decodeIfPresent([Int].self, forKey: .encKernelSizes) ?? [5, 3, 3, 3, 1]
        self.encDilations = try container.decodeIfPresent([Int].self, forKey: .encDilations) ?? [1, 2, 3, 4, 1]
        self.encAttentionChannels = try container.decodeIfPresent(Int.self, forKey: .encAttentionChannels) ?? 128
        self.encRes2netScale = try container.decodeIfPresent(Int.self, forKey: .encRes2netScale) ?? 8
        self.encSeChannels = try container.decodeIfPresent(Int.self, forKey: .encSeChannels) ?? 128
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
    }

    public init(
        melDim: Int = 128,
        encDim: Int = 1024,
        encChannels: [Int] = [512, 512, 512, 512, 1536],
        encKernelSizes: [Int] = [5, 3, 3, 3, 1],
        encDilations: [Int] = [1, 2, 3, 4, 1],
        encAttentionChannels: Int = 128,
        encRes2netScale: Int = 8,
        encSeChannels: Int = 128,
        sampleRate: Int = 24000
    ) {
        self.melDim = melDim
        self.encDim = encDim
        self.encChannels = encChannels
        self.encKernelSizes = encKernelSizes
        self.encDilations = encDilations
        self.encAttentionChannels = encAttentionChannels
        self.encRes2netScale = encRes2netScale
        self.encSeChannels = encSeChannels
        self.sampleRate = sampleRate
    }
}

// MARK: - Code Predictor Configuration

public struct Qwen3TTSCodePredictorConfig: Codable, Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var numCodeGroups: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numCodeGroups = "num_code_groups"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 2048
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 5
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.numCodeGroups = try container.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
    }

    public init(
        vocabSize: Int = 2048,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 5,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        numCodeGroups: Int = 16,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 65536
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.numCodeGroups = numCodeGroups
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }
}

// MARK: - RoPE Scaling Configuration

public struct Qwen3TTSRopeScaling: Codable, Sendable {
    public var interleaved: Bool
    public var mropeSection: [Int]
    public var ropeType: String

    enum CodingKeys: String, CodingKey {
        case interleaved
        case mropeSection = "mrope_section"
        case ropeType = "rope_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.interleaved = try container.decodeIfPresent(Bool.self, forKey: .interleaved) ?? true
        self.mropeSection = try container.decodeIfPresent([Int].self, forKey: .mropeSection) ?? [24, 20, 20]
        self.ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
    }

    public init(interleaved: Bool = true, mropeSection: [Int] = [24, 20, 20], ropeType: String = "default") {
        self.interleaved = interleaved
        self.mropeSection = mropeSection
        self.ropeType = ropeType
    }
}

// MARK: - Talker Configuration

public struct Qwen3TTSTalkerConfig: Codable, Sendable {
    // Model dimensions
    public var vocabSize: Int                    // 3072 - codec vocabulary size
    public var textVocabSize: Int                // 151936 - text vocabulary size
    public var hiddenSize: Int                   // 2048
    public var textHiddenSize: Int               // 2048
    public var intermediateSize: Int             // 6144
    public var perLayerIntermediateSizes: [Int]? // per-layer sizes (neuron pruning)
    public var numHiddenLayers: Int              // 28
    public var numAttentionHeads: Int            // 16
    public var numKeyValueHeads: Int             // 8 (GQA)
    public var headDim: Int                      // 128
    public var numCodeGroups: Int                // 16 codebooks

    // Normalization
    public var rmsNormEps: Float

    // RoPE configuration
    public var ropeTheta: Float
    public var ropeScaling: Qwen3TTSRopeScaling?
    public var maxPositionEmbeddings: Int

    // Codec special token IDs
    public var codecEosTokenId: Int
    public var codecThinkId: Int
    public var codecNothinkId: Int
    public var codecThinkBosId: Int
    public var codecThinkEosId: Int
    public var codecPadId: Int
    public var codecBosId: Int

    // Language ID mapping
    public var codecLanguageId: [String: Int]

    // Speaker configuration (for CustomVoice/Base models)
    // spkId: maps speaker name -> token ID
    public var spkId: [String: Int]?
    // spkIsDialect: maps speaker name -> dialect name (or nil if not a dialect)
    public var spkIsDialect: [String: DialectValue]?

    // Code Predictor
    public var codePredictorConfig: Qwen3TTSCodePredictorConfig?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case textVocabSize = "text_vocab_size"
        case hiddenSize = "hidden_size"
        case textHiddenSize = "text_hidden_size"
        case intermediateSize = "intermediate_size"
        case perLayerIntermediateSizes = "per_layer_intermediate_sizes"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numCodeGroups = "num_code_groups"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case codecEosTokenId = "codec_eos_token_id"
        case codecThinkId = "codec_think_id"
        case codecNothinkId = "codec_nothink_id"
        case codecThinkBosId = "codec_think_bos_id"
        case codecThinkEosId = "codec_think_eos_id"
        case codecPadId = "codec_pad_id"
        case codecBosId = "codec_bos_id"
        case codecLanguageId = "codec_language_id"
        case spkId = "spk_id"
        case spkIsDialect = "spk_is_dialect"
        case codePredictorConfig = "code_predictor_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 3072
        self.textVocabSize = try container.decodeIfPresent(Int.self, forKey: .textVocabSize) ?? 151936
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        self.textHiddenSize = try container.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? 2048
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.perLayerIntermediateSizes = try container.decodeIfPresent([Int].self, forKey: .perLayerIntermediateSizes)
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.numCodeGroups = try container.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        self.ropeScaling = try container.decodeIfPresent(Qwen3TTSRopeScaling.self, forKey: .ropeScaling)
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768

        // Codec special tokens
        self.codecEosTokenId = try container.decodeIfPresent(Int.self, forKey: .codecEosTokenId) ?? 2150
        self.codecThinkId = try container.decodeIfPresent(Int.self, forKey: .codecThinkId) ?? 2154
        self.codecNothinkId = try container.decodeIfPresent(Int.self, forKey: .codecNothinkId) ?? 2155
        self.codecThinkBosId = try container.decodeIfPresent(Int.self, forKey: .codecThinkBosId) ?? 2156
        self.codecThinkEosId = try container.decodeIfPresent(Int.self, forKey: .codecThinkEosId) ?? 2157
        self.codecPadId = try container.decodeIfPresent(Int.self, forKey: .codecPadId) ?? 2148
        self.codecBosId = try container.decodeIfPresent(Int.self, forKey: .codecBosId) ?? 2149

        self.codecLanguageId = try container.decodeIfPresent([String: Int].self, forKey: .codecLanguageId) ?? [
            "chinese": 2055,
            "english": 2050,
            "german": 2053,
            "italian": 2070,
            "portuguese": 2071,
            "spanish": 2054,
            "japanese": 2058,
            "korean": 2064,
            "french": 2061,
            "russian": 2069
        ]

        self.spkId = try container.decodeIfPresent([String: Int].self, forKey: .spkId)
        self.spkIsDialect = try container.decodeIfPresent([String: DialectValue].self, forKey: .spkIsDialect)
        self.codePredictorConfig = try container.decodeIfPresent(Qwen3TTSCodePredictorConfig.self, forKey: .codePredictorConfig)
    }
}

// MARK: - Speech Tokenizer Decoder Configuration

public struct Qwen3TTSTokenizerDecoderConfig: Codable, Sendable {
    public var latentDim: Int
    public var codebookDim: Int
    public var codebookSize: Int
    public var decoderDim: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var slidingWindow: Int
    public var numQuantizers: Int
    public var numSemanticQuantizers: Int
    public var semanticCodebookSize: Int
    public var upsampleRates: [Int]       // [8, 5, 4, 3] = 480
    public var upsamplingRatios: [Int]    // [2, 2] = 4, total = 1920
    public var vectorQuantizationHiddenDimension: Int
    public var layerScaleInitialScale: Float

    enum CodingKeys: String, CodingKey {
        case latentDim = "latent_dim"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case decoderDim = "decoder_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case numQuantizers = "num_quantizers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case semanticCodebookSize = "semantic_codebook_size"
        case upsampleRates = "upsample_rates"
        case upsamplingRatios = "upsampling_ratios"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
        case layerScaleInitialScale = "layer_scale_initial_scale"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.latentDim = try container.decodeIfPresent(Int.self, forKey: .latentDim) ?? 1024
        self.codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 512
        self.codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        self.decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1536
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1024
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 16
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 72
        self.numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 16
        self.numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        self.semanticCodebookSize = try container.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 4096
        self.upsampleRates = try container.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 5, 4, 3]
        self.upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [2, 2]
        self.vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 512
        self.layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
    }

    /// Total upsampling factor: 8*5*4*3 * 2*2 = 1920
    public var totalUpsample: Int {
        upsampleRates.reduce(1, *) * upsamplingRatios.reduce(1, *)
    }
}

// MARK: - Speech Tokenizer Encoder Configuration

public struct Qwen3TTSTokenizerEncoderConfig: Codable, Sendable {
    public var frameRate: Float
    public var audioChannels: Int
    public var codebookDim: Int
    public var codebookSize: Int
    public var compress: Int
    public var dilationGrowthRate: Int
    public var headDim: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var kernelSize: Int
    public var lastKernelSize: Int
    public var layerScaleInitialScale: Float
    public var maxPositionEmbeddings: Int
    public var numAttentionHeads: Int
    public var numFilters: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var numQuantizers: Int
    public var numResidualLayers: Int
    public var residualKernelSize: Int
    public var ropeTheta: Float
    public var samplingRate: Int
    public var slidingWindow: Int
    public var upsamplingRatios: [Int]
    public var useCausalConv: Bool
    public var useConvShortcut: Bool

    enum CodingKeys: String, CodingKey {
        case frameRate = "frame_rate"
        case audioChannels = "audio_channels"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case compress
        case dilationGrowthRate = "dilation_growth_rate"
        case headDim = "head_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numFilters = "num_filters"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numResidualLayers = "num_residual_layers"
        case residualKernelSize = "residual_kernel_size"
        case ropeTheta = "rope_theta"
        case samplingRate = "sampling_rate"
        case slidingWindow = "sliding_window"
        case upsamplingRatios = "upsampling_ratios"
        case useCausalConv = "use_causal_conv"
        case useConvShortcut = "use_conv_shortcut"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.frameRate = try container.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
        self.audioChannels = try container.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 1
        self.codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 256
        self.codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        self.compress = try container.decodeIfPresent(Int.self, forKey: .compress) ?? 2
        self.dilationGrowthRate = try container.decodeIfPresent(Int.self, forKey: .dilationGrowthRate) ?? 2
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        self.kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        self.lastKernelSize = try container.decodeIfPresent(Int.self, forKey: .lastKernelSize) ?? 3
        self.layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.numFilters = try container.decodeIfPresent(Int.self, forKey: .numFilters) ?? 64
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 32
        self.numResidualLayers = try container.decodeIfPresent(Int.self, forKey: .numResidualLayers) ?? 1
        self.residualKernelSize = try container.decodeIfPresent(Int.self, forKey: .residualKernelSize) ?? 3
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 250
        self.upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [8, 6, 5, 4]
        self.useCausalConv = try container.decodeIfPresent(Bool.self, forKey: .useCausalConv) ?? true
        self.useConvShortcut = try container.decodeIfPresent(Bool.self, forKey: .useConvShortcut) ?? false
    }

    public init(
        frameRate: Float = 12.5,
        audioChannels: Int = 1,
        codebookDim: Int = 256,
        codebookSize: Int = 2048,
        compress: Int = 2,
        dilationGrowthRate: Int = 2,
        headDim: Int = 64,
        hiddenSize: Int = 512,
        intermediateSize: Int = 2048,
        kernelSize: Int = 7,
        lastKernelSize: Int = 3,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        numAttentionHeads: Int = 8,
        numFilters: Int = 64,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 8,
        numQuantizers: Int = 32,
        numResidualLayers: Int = 1,
        residualKernelSize: Int = 3,
        ropeTheta: Float = 10000.0,
        samplingRate: Int = 24000,
        slidingWindow: Int = 250,
        upsamplingRatios: [Int] = [8, 6, 5, 4],
        useCausalConv: Bool = true,
        useConvShortcut: Bool = false
    ) {
        self.frameRate = frameRate
        self.audioChannels = audioChannels
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.compress = compress
        self.dilationGrowthRate = dilationGrowthRate
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.kernelSize = kernelSize
        self.lastKernelSize = lastKernelSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numAttentionHeads = numAttentionHeads
        self.numFilters = numFilters
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numResidualLayers = numResidualLayers
        self.residualKernelSize = residualKernelSize
        self.ropeTheta = ropeTheta
        self.samplingRate = samplingRate
        self.slidingWindow = slidingWindow
        self.upsamplingRatios = upsamplingRatios
        self.useCausalConv = useCausalConv
        self.useConvShortcut = useConvShortcut
    }
}

// MARK: - Speech Tokenizer Configuration

public struct Qwen3TTSTokenizerConfig: Codable, Sendable {
    public var encoderValidNumQuantizers: Int
    public var inputSampleRate: Int
    public var outputSampleRate: Int
    public var decodeUpsampleRate: Int       // 1920
    public var encodeDownsampleRate: Int     // 1920
    public var decoderConfig: Qwen3TTSTokenizerDecoderConfig?
    public var encoderConfig: Qwen3TTSTokenizerEncoderConfig?

    enum CodingKeys: String, CodingKey {
        case encoderValidNumQuantizers = "encoder_valid_num_quantizers"
        case inputSampleRate = "input_sample_rate"
        case outputSampleRate = "output_sample_rate"
        case decodeUpsampleRate = "decode_upsample_rate"
        case encodeDownsampleRate = "encode_downsample_rate"
        case decoderConfig = "decoder_config"
        case encoderConfig = "encoder_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.encoderValidNumQuantizers = try container.decodeIfPresent(Int.self, forKey: .encoderValidNumQuantizers) ?? 16
        self.inputSampleRate = try container.decodeIfPresent(Int.self, forKey: .inputSampleRate) ?? 24000
        self.outputSampleRate = try container.decodeIfPresent(Int.self, forKey: .outputSampleRate) ?? 24000
        self.decodeUpsampleRate = try container.decodeIfPresent(Int.self, forKey: .decodeUpsampleRate) ?? 1920
        self.encodeDownsampleRate = try container.decodeIfPresent(Int.self, forKey: .encodeDownsampleRate) ?? 1920
        self.decoderConfig = try container.decodeIfPresent(Qwen3TTSTokenizerDecoderConfig.self, forKey: .decoderConfig)
        self.encoderConfig = try container.decodeIfPresent(Qwen3TTSTokenizerEncoderConfig.self, forKey: .encoderConfig)
    }
}

// MARK: - Main Model Configuration

public struct Qwen3TTSModelConfig: Codable, Sendable {
    public var modelType: String
    public var talkerConfig: Qwen3TTSTalkerConfig?
    public var speakerEncoderConfig: Qwen3TTSSpeakerEncoderConfig?
    public var tokenizerType: String
    public var ttsModelSize: String          // "0b6" or "1b7"
    public var ttsModelType: String          // "base", "custom_voice", "voice_design"

    // TTS special token IDs
    public var imStartTokenId: Int           // 151644 - <|im_start|>
    public var imEndTokenId: Int             // 151645 - <|im_end|>
    public var ttsPadTokenId: Int            // 151671
    public var ttsBosTokenId: Int            // 151672
    public var ttsEosTokenId: Int            // 151673

    public var sampleRate: Int

    // Quantization
    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case talkerConfig = "talker_config"
        case speakerEncoderConfig = "speaker_encoder_config"
        case tokenizerType = "tokenizer_type"
        case ttsModelSize = "tts_model_size"
        case ttsModelType = "tts_model_type"
        case imStartTokenId = "im_start_token_id"
        case imEndTokenId = "im_end_token_id"
        case ttsPadTokenId = "tts_pad_token_id"
        case ttsBosTokenId = "tts_bos_token_id"
        case ttsEosTokenId = "tts_eos_token_id"
        case sampleRate = "sample_rate"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_tts"
        self.talkerConfig = try container.decodeIfPresent(Qwen3TTSTalkerConfig.self, forKey: .talkerConfig)
        self.speakerEncoderConfig = try container.decodeIfPresent(Qwen3TTSSpeakerEncoderConfig.self, forKey: .speakerEncoderConfig)
        self.tokenizerType = try container.decodeIfPresent(String.self, forKey: .tokenizerType) ?? "qwen3_tts_tokenizer_12hz"
        self.ttsModelSize = try container.decodeIfPresent(String.self, forKey: .ttsModelSize) ?? "1b7"
        self.ttsModelType = try container.decodeIfPresent(String.self, forKey: .ttsModelType) ?? "voice_design"

        self.imStartTokenId = try container.decodeIfPresent(Int.self, forKey: .imStartTokenId) ?? 151644
        self.imEndTokenId = try container.decodeIfPresent(Int.self, forKey: .imEndTokenId) ?? 151645
        self.ttsPadTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsPadTokenId) ?? 151671
        self.ttsBosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsBosTokenId) ?? 151672
        self.ttsEosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsEosTokenId) ?? 151673

        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000

        // Load quantization from base config
        let baseConfig = try? BaseConfiguration(from: decoder)
        self.quantization = baseConfig?.quantization
        self.perLayerQuantization = baseConfig?.perLayerQuantization
    }

    /// Convenience access to talker's codec EOS token ID
    public var codecEosTokenId: Int {
        talkerConfig?.codecEosTokenId ?? 2150
    }

    /// Convenience access to MRoPE section
    public var mropeSection: [Int] {
        talkerConfig?.ropeScaling?.mropeSection ?? [24, 20, 20]
    }
}
