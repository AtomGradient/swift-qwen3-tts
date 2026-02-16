//
//  CodePredictor.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS Code Predictor
//  Predicts tokens for codebooks 2-16 (15 additional codebooks)
//
//  Ported from Python mlx-audio implementation
//

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN
import MLXLMCommon

// MARK: - Standard RoPE (for Code Predictor)

/// Standard Rotary Position Embedding (not MRoPE)
public class CodePredictorRotaryEmbedding: Module {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float

    private var _invFreq: MLXArray

    public init(
        dim: Int,
        maxPositionEmbeddings: Int = 65536,
        base: Float = 1_000_000.0
    ) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base

        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self._invFreq = 1.0 / pow(MLXArray(base), indices / Float(dim))
    }

    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        // positionIds: [batch, seq_len]
        let posFloat = positionIds.asType(.float32)

        // freqs = outer(positionIds, invFreq)
        let invFreqExpanded = _invFreq.reshaped([1, 1, -1])  // [1, 1, dim/2]
        let posExpanded = posFloat.expandedDimensions(axis: -1)  // [batch, seq_len, 1]

        let freqs = posExpanded * invFreqExpanded  // [batch, seq_len, dim/2]

        // Expand to full dim
        let emb = MLX.concatenated([freqs, freqs], axis: -1)  // [batch, seq_len, dim]

        let cos = MLX.cos(emb).asType(x.dtype)
        let sin = MLX.sin(emb).asType(x.dtype)

        return (cos, sin)
    }
}

// MARK: - Code Predictor Attention

public class CodePredictorAttention: Module {
    let config: Qwen3TTSCodePredictorConfig
    let layerIdx: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    public init(_ config: Qwen3TTSCodePredictorConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        q = q.reshaped(B, L, numHeads, headDim)
        k = k.reshaped(B, L, numKVHeads, headDim)
        v = v.reshaped(B, L, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(outputReshaped)
    }
}

// MARK: - Code Predictor MLP

public class CodePredictorMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: Qwen3TTSCodePredictorConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Code Predictor Decoder Layer

public class CodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CodePredictorAttention
    let mlp: CodePredictorMLP

    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(_ config: Qwen3TTSCodePredictorConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = CodePredictorAttention(config, layerIdx: layerIdx)
        self.mlp = CodePredictorMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        var residual = x
        var h = inputLayernorm(x)
        h = selfAttn(h, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        h = residual + h

        residual = h
        h = postAttentionLayernorm(h)
        h = mlp(h)
        h = residual + h

        return h
    }
}

// MARK: - Code Predictor Model

/// Internal transformer model for code prediction
public class CodePredictorModel: Module {
    let config: Qwen3TTSCodePredictorConfig
    let talkerHiddenSize: Int

    /// 15 codec embeddings for codebooks 2-16
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]

    let layers: [CodePredictorDecoderLayer]
    let norm: RMSNorm
    let rotaryEmb: CodePredictorRotaryEmbedding

    public init(_ config: Qwen3TTSCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.talkerHiddenSize = talkerHiddenSize

        // 15 codec embeddings (for codebooks 2-16)
        self._codecEmbedding.wrappedValue = (0..<(config.numCodeGroups - 1)).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }

        // 5 transformer layers
        self.layers = (0..<config.numHiddenLayers).map { i in
            CodePredictorDecoderLayer(config, layerIdx: i)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Standard RoPE (not MRoPE)
        self.rotaryEmb = CodePredictorRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        var h = inputsEmbeds

        let seqLen = h.dim(1)
        let batchSize = h.dim(0)
        let offset = cache?.first?.offset ?? 0

        // Position IDs
        let positions = MLXArray(Array(offset..<(offset + seqLen)).map { Int32($0) })
        let pos = MLX.broadcast(positions.expandedDimensions(axis: 0), to: [batchSize, seqLen])

        // Position embeddings
        let posEmb = rotaryEmb(h, positionIds: pos)

        // Create causal mask if needed (matching Python behavior)
        var effectiveMask: MLXArray? = nil
        if seqLen > 1 {
            let ones = MLXArray.ones([seqLen, seqLen])
            let mask = MLX.triu(ones, k: 1)
            effectiveMask = MLX.where(mask .== 1, MLXArray(-Float.infinity), MLXArray(Float(0)))
            effectiveMask = effectiveMask!.asType(h.dtype)
        }

        // Process through layers
        for (i, layer) in layers.enumerated() {
            h = layer(h, positionEmbeddings: posEmb, mask: effectiveMask, cache: cache?[i])
        }

        return norm(h)
    }

    public func makeCache() -> [KVCache] {
        return (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }
    }
}

// MARK: - Qwen3 TTS Code Predictor

/// Main Code Predictor for predicting tokens of codebooks 2-16
public class Qwen3TTSCodePredictor: Module {
    let config: Qwen3TTSCodePredictorConfig
    let numCodeGroups: Int

    /// Projection from talker hidden size to code predictor hidden size
    @ModuleInfo(key: "small_to_mtp_projection") var smallToMtpProjection: Linear?

    let model: CodePredictorModel

    /// 15 LM heads (one for each codebook 2-16)
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    public init(_ config: Qwen3TTSCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.numCodeGroups = config.numCodeGroups

        // Dimension projection if needed
        if config.hiddenSize != talkerHiddenSize {
            self._smallToMtpProjection.wrappedValue = Linear(
                talkerHiddenSize, config.hiddenSize, bias: true
            )
        }

        self.model = CodePredictorModel(config, talkerHiddenSize: talkerHiddenSize)

        // 15 LM heads for codebooks 2-16
        self._lmHead.wrappedValue = (0..<(config.numCodeGroups - 1)).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    /// Access to codec embeddings
    public var codecEmbedding: [Embedding] {
        return model.codecEmbedding
    }

    /// Forward pass
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings
    ///   - cache: KV cache
    ///   - generationStep: Which codebook (0-14) to predict
    /// - Returns: (logits, updated cache, next generation step)
    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        cache: [KVCache]?,
        generationStep: Int
    ) -> (logits: MLXArray, cache: [KVCache]?, nextStep: Int) {
        var h = inputsEmbeds

        // Project dimensions if needed
        if let projection = smallToMtpProjection {
            h = projection(h)
        }

        // Run through model
        h = model(h, cache: cache)

        // Get logits for current codebook
        let logits = lmHead[generationStep](h)

        return (logits, cache, generationStep + 1)
    }

    public func makeCache() -> [KVCache] {
        return model.makeCache()
    }
}
