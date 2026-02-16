//
//  Talker.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS Talker Model
//  Main autoregressive transformer for codec token generation
//
//  Ported from Python mlx-audio implementation
//

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN
import MLXLMCommon

// MARK: - MRoPE (Multimodal Rotary Position Embedding)

/// Multimodal RoPE with interleaved frequency combination
/// Combines 3D positions (temporal, height, width) with interleaved pattern
public class TalkerRotaryEmbedding: Module {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let mropeSection: [Int]  // [24, 20, 20] = 64 (head_dim / 2)

    private var _invFreq: MLXArray

    public init(
        dim: Int,
        maxPositionEmbeddings: Int = 32768,
        base: Float = 1_000_000.0,
        mropeSection: [Int] = [24, 20, 20]
    ) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.mropeSection = mropeSection

        // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self._invFreq = 1.0 / pow(MLXArray(base), indices / Float(dim))
    }

    /// Apply interleaved MRoPE combination
    /// Rearranges from [TTT...HHH...WWW] to [THTHWHTHW...TT]
    private func applyInterleavedMrope(_ freqs: MLXArray) -> MLXArray {
        // freqs: [3, batch, seq_len, head_dim/2]
        let headDimHalf = freqs.dim(-1)

        let freqsT = freqs[0]  // Temporal: [batch, seq_len, head_dim/2]
        let freqsH = freqs[1]  // Height
        let freqsW = freqs[2]  // Width

        // Create interleaved pattern using loops
        // For mrope_section = [24, 20, 20]:
        // - First 60 positions (20*3): interleaved T,H,W pattern
        // - Remaining 4 positions: T only (24 - 20 = 4)

        let hLength = mropeSection[1] * 3  // 60
        let wLength = mropeSection[2] * 3  // 60

        // Create masks for H and W positions
        var hMaskVals = [Bool]()
        var wMaskVals = [Bool]()
        for i in 0..<headDimHalf {
            let mod3 = i % 3
            hMaskVals.append(mod3 == 1 && i < hLength)
            wMaskVals.append(mod3 == 2 && i < wLength)
        }
        let hMask = MLXArray(hMaskVals)
        let wMask = MLXArray(wMaskVals)

        // Interleave: start with T, replace with H where hMask, replace with W where wMask
        var combined = freqsT
        combined = MLX.where(hMask, freqsH, combined)
        combined = MLX.where(wMask, freqsW, combined)

        return combined
    }

    /// Compute position embeddings
    /// - Parameters:
    ///   - x: Input tensor for dtype reference
    ///   - positionIds: Position IDs, shape [3, batch, seq_len] or [batch, seq_len]
    /// - Returns: (cos, sin) embeddings
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        var pos = positionIds

        // Expand 2D positions to 3D if needed
        if pos.ndim == 2 {
            // [batch, seq_len] -> [3, batch, seq_len]
            pos = MLX.stacked([pos, pos, pos], axis: 0)
        }

        // pos: [3, batch, seq_len]
        // invFreq: [head_dim/2]

        // Compute frequencies for each dimension
        // freqs = invFreq @ pos -> [3, head_dim/2, batch, seq_len]
        let posFloat = pos.asType(.float32)
        let invFreqExpanded = _invFreq.reshaped([1, -1, 1, 1])  // [1, head_dim/2, 1, 1]
        let posExpanded = posFloat.reshaped([3, 1, pos.dim(1), pos.dim(2)])  // [3, 1, batch, seq_len]

        var freqs = invFreqExpanded * posExpanded  // [3, head_dim/2, batch, seq_len]
        freqs = freqs.transposed(0, 2, 3, 1)  // [3, batch, seq_len, head_dim/2]

        // Apply interleaved MRoPE
        freqs = applyInterleavedMrope(freqs)  // [batch, seq_len, head_dim/2]

        // Expand to full head_dim
        let emb = MLX.concatenated([freqs, freqs], axis: -1)  // [batch, seq_len, head_dim]

        let cos = MLX.cos(emb).asType(x.dtype)
        let sin = MLX.sin(emb).asType(x.dtype)

        return (cos, sin)
    }
}

/// Rotate half of the hidden dimensions
/// Python: rotate_half returns [-x2, x1]
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

/// Apply rotary position embedding to query and key tensors
/// - Parameters:
///   - q: Query tensor [batch, heads, seq_len, head_dim]
///   - k: Key tensor [batch, heads, seq_len, head_dim]
///   - cos: Cosine embeddings [batch, seq_len, head_dim]
///   - sin: Sine embeddings [batch, seq_len, head_dim]
/// - Returns: (rotated_q, rotated_k)
func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray,
    cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    // Expand cos/sin for heads dimension
    let cosExpanded = cos.expandedDimensions(axis: 1)  // [batch, 1, seq_len, head_dim]
    let sinExpanded = sin.expandedDimensions(axis: 1)

    // Apply rotation: q_embed = q * cos + rotate_half(q) * sin
    let qRotated = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kRotated = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)

    return (qRotated, kRotated)
}

// MARK: - Talker Attention

/// Grouped Query Attention with QK Normalization
public class TalkerAttention: Module {
    let config: Qwen3TTSTalkerConfig
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

    public init(_ config: Qwen3TTSTalkerConfig, layerIdx: Int) {
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

        // QK Normalization (Qwen3 specific)
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

        // Project Q, K, V
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        // Reshape to [B, L, heads, head_dim]
        q = q.reshaped(B, L, numHeads, headDim)
        k = k.reshaped(B, L, numKVHeads, headDim)
        v = v.reshaped(B, L, numKVHeads, headDim)

        // Apply QK Normalization (before RoPE)
        q = qNorm(q)
        k = kNorm(k)

        // Transpose to [B, heads, L, head_dim]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Apply MRoPE
        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        // Update KV cache
        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Scaled dot-product attention with GQA
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        // Reshape back and project output
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(outputReshaped)
    }
}

// MARK: - Activation Profiler

/// Records per-neuron activation magnitudes across forward passes for pruning analysis.
/// Uses MLX arrays for accumulation during generation, converts to Swift only at flush time.
/// This avoids disrupting MLX's lazy evaluation graph during autoregressive decoding.
public final class ActivationProfiler: @unchecked Sendable {
    public static let shared = ActivationProfiler()

    public private(set) var isEnabled = false
    public private(set) var numLayers = 0
    public private(set) var intermediateSize = 0
    public private(set) var runCount = 0

    /// Global max activations across all runs: [layer][neuron]
    private var globalMaxActivations: [[Float]] = []
    /// Sum of per-run max activations (for computing mean): [layer][neuron]
    private var globalSumActivations: [[Float]] = []

    /// Current run's per-layer max (MLX arrays, accumulated lazily during generation)
    private var currentRunMax: [MLXArray?] = []

    private init() {}

    /// Enable profiling for a model with given dimensions.
    public func enable(numLayers: Int, intermediateSize: Int) {
        self.numLayers = numLayers
        self.intermediateSize = intermediateSize
        self.isEnabled = true
        self.runCount = 0
        self.globalMaxActivations = Array(repeating: Array(repeating: Float(0), count: intermediateSize), count: numLayers)
        self.globalSumActivations = Array(repeating: Array(repeating: Float(0), count: intermediateSize), count: numLayers)
        self.currentRunMax = Array(repeating: nil, count: numLayers)
    }

    public func disable() {
        isEnabled = false
    }

    /// Record activation magnitudes for a layer during a single token step.
    /// Does NOT call eval() — keeps everything in MLX's lazy graph.
    public func record(layerIdx: Int, activation: MLXArray) {
        guard isEnabled, layerIdx < numLayers else { return }

        // max(|activation|) over batch and seq dims → [intermediate_size]
        let maxPerNeuron = abs(activation).max(axes: [0, 1])  // [intermediate_size]

        if let existing = currentRunMax[layerIdx] {
            currentRunMax[layerIdx] = maximum(existing, maxPerNeuron)
        } else {
            currentRunMax[layerIdx] = maxPerNeuron
        }
    }

    /// Flush current run's accumulated activations into global stats.
    /// Call this AFTER each complete generation, outside the autoregressive loop.
    public func flushRun() {
        guard isEnabled else { return }

        // Evaluate all layers' max activations at once
        let layersToEval = currentRunMax.compactMap { $0 }
        eval(layersToEval)

        for layer in 0..<numLayers {
            guard let runMax = currentRunMax[layer] else { continue }
            let values = runMax.asArray(Float.self)
            guard values.count == intermediateSize else { continue }

            for i in 0..<intermediateSize {
                globalMaxActivations[layer][i] = max(globalMaxActivations[layer][i], values[i])
                globalSumActivations[layer][i] += values[i]
            }
        }

        runCount += 1
        // Reset for next run
        currentRunMax = Array(repeating: nil, count: numLayers)
    }

    /// Print summary statistics.
    public func printSummary(thresholds: [Float] = [0.01, 0.05, 0.1, 0.5, 1.0]) {
        print("\n" + String(repeating: "=", count: 70))
        print("ACTIVATION PROFILING RESULTS (\(runCount) runs)")
        print(String(repeating: "=", count: 70))

        // Header
        var header = " Layer |"
        for t in thresholds {
            header += String(format: " <%5.2f", t)
        }
        header += " | Active |  Dead%"
        print(header)
        print(String(repeating: "-", count: 70))

        var totalPrunable = Array(repeating: 0, count: thresholds.count)

        for layer in 0..<numLayers {
            let maxActs = globalMaxActivations[layer]
            var counts: [Int] = []
            for t in thresholds {
                let c = maxActs.filter { $0 < t }.count
                counts.append(c)
            }

            let active = intermediateSize - counts[counts.count - 1]
            let deadPct = Float(counts[counts.count - 1]) / Float(intermediateSize) * 100

            var line = String(format: "  L%3d  |", layer)
            for c in counts {
                line += String(format: " %5d", c)
            }
            line += String(format: " | %6d | %5.1f%%", active, deadPct)
            print(line)

            for (i, c) in counts.enumerated() {
                totalPrunable[i] += c
            }
        }

        print(String(repeating: "-", count: 70))
        var avgLine = String(format: "  AVG  |")
        for tp in totalPrunable {
            avgLine += String(format: " %5d", tp / numLayers)
        }
        let avgDeadPct = Float(totalPrunable[totalPrunable.count - 1]) / Float(numLayers * intermediateSize) * 100
        avgLine += String(format: " |        | %5.1f%%", avgDeadPct)
        print(avgLine)
        print()
    }

    /// Save results to JSON file.
    public func saveJSON(to path: String) throws {
        var layerData: [[String: Any]] = []
        for layer in 0..<numLayers {
            let meanActs = runCount > 0
                ? globalSumActivations[layer].map { $0 / Float(runCount) }
                : globalSumActivations[layer]
            layerData.append([
                "layer": layer,
                "max_activations": globalMaxActivations[layer],
                "mean_activations": meanActs
            ])
        }

        let result: [String: Any] = [
            "num_layers": numLayers,
            "intermediate_size": intermediateSize,
            "run_count": runCount,
            "layers": layerData
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: path))
        print("Activation profile saved to: \(path)")
    }
}

// MARK: - Talker MLP (SwiGLU)

/// SwiGLU MLP: down(silu(gate(x)) * up(x))
public class TalkerMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    var layerIdx: Int = -1

    public init(_ config: Qwen3TTSTalkerConfig, intermediateSize: Int? = nil) {
        let hiddenSize = config.hiddenSize
        let iSize = intermediateSize ?? config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, iSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, iSize, bias: false)
        self._downProj.wrappedValue = Linear(iSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gatedActivation = silu(gateProj(x)) * upProj(x)

        // Record activation if profiling is enabled
        let profiler = ActivationProfiler.shared
        if profiler.isEnabled && layerIdx >= 0 {
            profiler.record(layerIdx: layerIdx, activation: gatedActivation)
        }

        return downProj(gatedActivation)
    }
}

// MARK: - Talker Decoder Layer

/// Pre-norm decoder layer with attention and MLP
public class TalkerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TalkerAttention
    let mlp: TalkerMLP

    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(_ config: Qwen3TTSTalkerConfig, layerIdx: Int, intermediateSize: Int? = nil) {
        self._selfAttn.wrappedValue = TalkerAttention(config, layerIdx: layerIdx)
        let mlpModule = TalkerMLP(config, intermediateSize: intermediateSize)
        mlpModule.layerIdx = layerIdx
        self.mlp = mlpModule
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        // Pre-norm architecture
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

// MARK: - ResizeMLP (Text Projection)

/// MLP for projecting text embeddings to hidden size
public class ResizeMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    public init(inputSize: Int, intermediateSize: Int, outputSize: Int, bias: Bool = true) {
        self._fc1.wrappedValue = Linear(inputSize, intermediateSize, bias: bias)
        self._fc2.wrappedValue = Linear(intermediateSize, outputSize, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fc2(silu(fc1(x)))
    }
}

// MARK: - Qwen3 TTS Talker Model

/// Main Talker transformer model
public class Qwen3TTSTalkerModel: Module {
    let config: Qwen3TTSTalkerConfig

    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding

    /// Optional token map for pruned vocabulary: maps original token ID → compact index.
    /// Loaded manually from weights (not a Module parameter).
    var textTokenMap: MLXArray?

    let layers: [TalkerDecoderLayer]
    let norm: RMSNorm
    let rotaryEmb: TalkerRotaryEmbedding

    public init(_ config: Qwen3TTSTalkerConfig) {
        self.config = config

        // Dual embeddings
        self._codecEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._textEmbedding.wrappedValue = Embedding(embeddingCount: config.textVocabSize, dimensions: config.textHiddenSize)

        // Transformer layers (support per-layer intermediate sizes for neuron pruning)
        let perLayerSizes = config.perLayerIntermediateSizes
        self.layers = (0..<config.numHiddenLayers).map { i in
            let iSize = perLayerSizes?[i]
            return TalkerDecoderLayer(config, layerIdx: i, intermediateSize: iSize)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // MRoPE
        let mropeSection = config.ropeScaling?.mropeSection ?? [24, 20, 20]
        self.rotaryEmb = TalkerRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta,
            mropeSection: mropeSection
        )
    }

    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray?,
        mask: MLXArray?,
        cache: [KVCache]?
    ) -> MLXArray {
        var h = inputsEmbeds

        // Compute position IDs if not provided
        let seqLen = h.dim(1)
        let batchSize = h.dim(0)
        let offset = cache?.first?.offset ?? 0

        let pos: MLXArray
        if let positionIds = positionIds {
            pos = positionIds
        } else {
            // Default: simple sequential positions
            let positions = MLXArray(Array(offset..<(offset + seqLen)).map { Int32($0) })
            pos = MLX.broadcast(positions.expandedDimensions(axis: 0), to: [batchSize, seqLen])
        }

        // Compute position embeddings
        let posEmb = rotaryEmb(h, positionIds: pos)

        // Create causal mask if not provided (matching Python behavior)
        var effectiveMask = mask
        if mask == nil && seqLen > 1 {
            // Create additive causal mask: lower triangular with -inf for masked positions
            // Matches nn.MultiHeadAttention.create_additive_causal_mask in Python
            let ones = MLXArray.ones([seqLen, seqLen])
            let mask = MLX.triu(ones, k: 1)  // Upper triangular (positions to mask)
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

// MARK: - Qwen3 TTS Talker For Conditional Generation

/// Wrapper model with text projection, codec head, and code predictor
public class Qwen3TTSTalkerForConditionalGeneration: Module {
    let config: Qwen3TTSTalkerConfig

    let model: Qwen3TTSTalkerModel

    @ModuleInfo(key: "text_projection") var textProjection: ResizeMLP
    @ModuleInfo(key: "codec_head") var codecHead: Linear
    @ModuleInfo(key: "code_predictor") var codePredictor: Qwen3TTSCodePredictor?

    public init(_ config: Qwen3TTSTalkerConfig) {
        self.config = config

        self.model = Qwen3TTSTalkerModel(config)

        // Text projection MLP
        self._textProjection.wrappedValue = ResizeMLP(
            inputSize: config.textHiddenSize,
            intermediateSize: config.textHiddenSize,
            outputSize: config.hiddenSize,
            bias: true
        )

        // Codec output head (first codebook)
        self._codecHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        // Code predictor (remaining 15 codebooks)
        if let cpConfig = config.codePredictorConfig {
            self._codePredictor.wrappedValue = Qwen3TTSCodePredictor(cpConfig, talkerHiddenSize: config.hiddenSize)
        }
    }

    /// Get codec embedding layer (first codebook)
    public func getInputEmbeddings() -> Embedding {
        return model.codecEmbedding
    }

    /// Get text embedding layer
    public func getTextEmbeddings() -> Embedding {
        return model.textEmbedding
    }

    /// Embed text token IDs, applying token map if vocabulary is pruned.
    /// Use this instead of `getTextEmbeddings()(ids)` for token-map-aware lookup.
    public func embedText(_ ids: MLXArray) -> MLXArray {
        if let tokenMap = model.textTokenMap {
            let mappedIds = tokenMap[ids]
            return model.textEmbedding(mappedIds)
        }
        return model.textEmbedding(ids)
    }

    /// Forward pass
    /// - Returns: (logits, hidden_states)
    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> (logits: MLXArray, hiddenStates: MLXArray) {
        let hiddenStates = model(inputsEmbeds, positionIds: positionIds, mask: mask, cache: cache)
        let logits = codecHead(hiddenStates)
        return (logits, hiddenStates)
    }

    public func makeCache() -> [KVCache] {
        return model.makeCache()
    }
}

