//
//  SpeechTokenizerEncoder.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Speech Tokenizer Encoder for Qwen3-TTS Voice Cloning (ICL mode)
//  Ported from Python mlx-audio Mimi implementation
//
//  Architecture: SeanetEncoder -> ProjectedTransformer -> ConvDownsample1d -> SplitRVQ
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

// MARK: - Configuration

/// Configuration for Seanet encoder
public struct SeanetEncoderConfig {
    public var dimension: Int          // 512 - output dimension
    public var channels: Int           // 1 - input audio channels
    public var causal: Bool            // true
    public var nfilters: Int           // 64 - base number of filters
    public var nresidualLayers: Int    // 1 - residual layers per block
    public var ratios: [Int]           // [8, 6, 5, 4] - downsampling ratios
    public var ksize: Int              // 7 - kernel size
    public var residualKsize: Int      // 3 - residual kernel size
    public var lastKsize: Int          // 3 - final conv kernel size
    public var dilationBase: Int       // 2 - dilation growth rate
    public var padMode: String         // "constant"
    public var trueSkip: Bool          // true - use true skip connection
    public var compress: Int           // 2 - compression factor

    public init(
        dimension: Int = 512,
        channels: Int = 1,
        causal: Bool = true,
        nfilters: Int = 64,
        nresidualLayers: Int = 1,
        ratios: [Int] = [8, 6, 5, 4],
        ksize: Int = 7,
        residualKsize: Int = 3,
        lastKsize: Int = 3,
        dilationBase: Int = 2,
        padMode: String = "constant",
        trueSkip: Bool = true,
        compress: Int = 2
    ) {
        self.dimension = dimension
        self.channels = channels
        self.causal = causal
        self.nfilters = nfilters
        self.nresidualLayers = nresidualLayers
        self.ratios = ratios
        self.ksize = ksize
        self.residualKsize = residualKsize
        self.lastKsize = lastKsize
        self.dilationBase = dilationBase
        self.padMode = padMode
        self.trueSkip = trueSkip
        self.compress = compress
    }
}

/// Configuration for encoder Transformer
public struct EncoderTransformerConfig {
    public var dModel: Int             // 512
    public var numHeads: Int           // 8
    public var numLayers: Int          // 8
    public var causal: Bool            // true
    public var layerScale: Float?      // 0.01
    public var context: Int            // 250 - sliding window
    public var maxPeriod: Int          // 10000 - rope theta
    public var maxSeqLen: Int          // 8000
    public var kvRepeat: Int           // 1
    public var dimFeedforward: Int     // 2048
    public var convLayout: Bool        // true - input is NCL format

    public var headDim: Int { dModel / numHeads }

    public init(
        dModel: Int = 512,
        numHeads: Int = 8,
        numLayers: Int = 8,
        causal: Bool = true,
        layerScale: Float? = 0.01,
        context: Int = 250,
        maxPeriod: Int = 10000,
        maxSeqLen: Int = 8000,
        kvRepeat: Int = 1,
        dimFeedforward: Int = 2048,
        convLayout: Bool = true
    ) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.causal = causal
        self.layerScale = layerScale
        self.context = context
        self.maxPeriod = maxPeriod
        self.maxSeqLen = maxSeqLen
        self.kvRepeat = kvRepeat
        self.dimFeedforward = dimFeedforward
        self.convLayout = convLayout
    }
}

// MARK: - Helper Functions

/// Get extra padding needed for causal conv to produce exact output frames
private func getExtraPaddingForConv1d(length: Int, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    let nframes = Float(max(length + paddingTotal - ksize, 0)) / Float(stride) + 1.0
    let idealLength = (Int(ceil(nframes)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLength - length)
}

// MARK: - Streamable Conv1d

/// Streamable 1D convolution with causal padding support
/// Input/Output format: NCL [batch, channels, time]
public class StreamableConv1d: Module {
    let causal: Bool
    let padMode: String
    let ksize: Int
    let stride: Int
    let dilation: Int
    let outChannels: Int

    @ModuleInfo(key: "conv") var conv: NormConv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        dilation: Int,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = true,
        padMode: String = "constant"
    ) {
        self.causal = causal
        self.padMode = padMode
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.outChannels = outChannels

        self._conv.wrappedValue = NormConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            ksize: ksize,
            stride: stride,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let effectiveKsize = (ksize - 1) * dilation + 1
        let paddingTotal = effectiveKsize - stride
        let extraPadding = getExtraPaddingForConv1d(
            length: xs.dim(-1),
            ksize: effectiveKsize,
            stride: stride,
            paddingTotal: paddingTotal
        )

        let paddingLeft: Int
        let paddingRight: Int
        if causal {
            paddingLeft = paddingTotal
            paddingRight = extraPadding
        } else {
            paddingRight = paddingTotal / 2 + extraPadding
            paddingLeft = paddingTotal - paddingTotal / 2
        }

        // Pad the time dimension (last axis)
        let padded = MLX.padded(xs, widths: [[0, 0], [0, 0], [paddingLeft, paddingRight]])
        return conv(padded)
    }
}

/// Normalized Conv1d wrapper
public class NormConv1d: Module {
    let stride: Int
    let dilation: Int

    @ModuleInfo(key: "conv") var innerConv: EncoderConv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.dilation = dilation
        self._innerConv.wrappedValue = EncoderConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            ksize: ksize,
            stride: stride,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return innerConv(xs)
    }
}

/// Low-level Conv1d for encoder (NCL format)
public class EncoderConv1d: Module {
    let stride: Int
    let dilation: Int
    let groups: Int
    let hasBias: Bool

    var weight: MLXArray
    var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.hasBias = bias

        // Weight shape: [outChannels, ksize, inChannels/groups] (MLX format)
        let scale = Float(1.0) / Float(inChannels * ksize)
        self.weight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, ksize, inChannels / groups]
        )

        if bias {
            self.bias = MLXArray.zeros([outChannels])
        } else {
            self.bias = nil
        }
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        // xs: NCL [batch, channels, time]
        // MLX conv1d expects NLC [batch, time, channels]
        let xNLC = xs.transposed(0, 2, 1)

        var y = MLX.conv1d(
            xNLC,
            weight,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups
        )

        if let b = bias {
            y = y + b
        }

        // Back to NCL
        return y.transposed(0, 2, 1)
    }
}

// MARK: - Seanet Residual Block

/// Residual block for Seanet encoder
public class SeanetResnetBlock: Module {
    let trueSkip: Bool

    @ModuleInfo(key: "block") var block: [StreamableConv1d]
    @ModuleInfo(key: "shortcut") var shortcut: StreamableConv1d?

    public init(_ cfg: SeanetEncoderConfig, dim: Int, ksizesAndDilations: [(Int, Int)]) {
        self.trueSkip = cfg.trueSkip
        let hidden = dim / cfg.compress

        var blockList: [StreamableConv1d] = []
        for (i, (ksize, dilation)) in ksizesAndDilations.enumerated() {
            let inChannels = i == 0 ? dim : hidden
            let outChannels = i == ksizesAndDilations.count - 1 ? dim : hidden
            blockList.append(StreamableConv1d(
                inChannels: inChannels,
                outChannels: outChannels,
                ksize: ksize,
                stride: 1,
                dilation: dilation,
                groups: 1,
                bias: true,
                causal: cfg.causal,
                padMode: cfg.padMode
            ))
        }
        self._block.wrappedValue = blockList

        if cfg.trueSkip {
            self._shortcut.wrappedValue = nil
        } else {
            self._shortcut.wrappedValue = StreamableConv1d(
                inChannels: dim,
                outChannels: dim,
                ksize: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias: true,
                causal: cfg.causal,
                padMode: cfg.padMode
            )
        }
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let residual = xs
        var out = xs

        for b in block {
            out = elu(out, alpha: 1.0)
            out = b(out)
        }

        if let sc = shortcut {
            return out + sc(residual)
        } else {
            return out + residual
        }
    }
}

// MARK: - Encoder Layer

/// Encoder layer with residual blocks and downsampling
public class SeanetEncoderLayer: Module {
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: StreamableConv1d

    public init(_ cfg: SeanetEncoderConfig, ratio: Int, mult: Int) {
        var residualList: [SeanetResnetBlock] = []
        var dilation = 1

        for _ in 0..<cfg.nresidualLayers {
            residualList.append(SeanetResnetBlock(
                cfg,
                dim: mult * cfg.nfilters,
                ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)]
            ))
            dilation *= cfg.dilationBase
        }
        self._residuals.wrappedValue = residualList

        self._downsample.wrappedValue = StreamableConv1d(
            inChannels: mult * cfg.nfilters,
            outChannels: mult * cfg.nfilters * 2,
            ksize: ratio * 2,
            stride: ratio,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: true,
            padMode: cfg.padMode
        )
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var out = xs
        for r in residuals {
            out = r(out)
        }
        return downsample(elu(out, alpha: 1.0))
    }
}

// MARK: - Seanet Encoder

/// Main Seanet encoder
public class SeanetEncoder: Module {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [SeanetEncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    public init(_ cfg: SeanetEncoderConfig) {
        var mult = 1

        self._initConv1d.wrappedValue = StreamableConv1d(
            inChannels: cfg.channels,
            outChannels: mult * cfg.nfilters,
            ksize: cfg.ksize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: cfg.causal,
            padMode: cfg.padMode
        )

        var layerList: [SeanetEncoderLayer] = []
        for ratio in cfg.ratios.reversed() {
            layerList.append(SeanetEncoderLayer(cfg, ratio: ratio, mult: mult))
            mult *= 2
        }
        self._layers.wrappedValue = layerList

        self._finalConv1d.wrappedValue = StreamableConv1d(
            inChannels: mult * cfg.nfilters,
            outChannels: cfg.dimension,
            ksize: cfg.lastKsize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: cfg.causal,
            padMode: cfg.padMode
        )
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var out = initConv1d(xs)
        for layer in layers {
            out = layer(out)
        }
        out = elu(out, alpha: 1.0)
        return finalConv1d(out)
    }
}

// MARK: - Transformer Components

/// Layer scale for residual connections
public class EncoderLayerScale: Module {
    var scale: MLXArray

    public init(dim: Int, initialScale: Float = 0.01) {
        self.scale = MLXArray.ones([dim]) * initialScale
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return xs * scale
    }
}

/// Identity module
public class EncoderIdentity: Module {
    public override init() {
        super.init()
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return xs
    }
}

/// Attention for encoder transformer
public class EncoderAttention: Module {
    let cfg: EncoderTransformerConfig
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    var rope: RoPE?

    public init(_ cfg: EncoderTransformerConfig) {
        self.cfg = cfg
        self.scale = pow(Float(cfg.headDim), -0.5)

        let numKV = cfg.numHeads / cfg.kvRepeat

        self._qProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: false)
        self._kProj.wrappedValue = Linear(cfg.dModel, numKV * cfg.headDim, bias: false)
        self._vProj.wrappedValue = Linear(cfg.dModel, numKV * cfg.headDim, bias: false)
        self._oProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: false)

        self.rope = RoPE(dimensions: cfg.headDim, traditional: false, base: Float(cfg.maxPeriod))
    }

    public func callAsFunction(_ xs: MLXArray, cache: (any KVCache)?, mask: MLXArray?) -> MLXArray {
        let (b, t, _) = (xs.dim(0), xs.dim(1), xs.dim(2))
        let offset = cache?.offset ?? 0

        var q = qProj(xs).reshaped([b, t, cfg.numHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        var k = kProj(xs).reshaped([b, t, cfg.numHeads / cfg.kvRepeat, cfg.headDim]).transposed(0, 2, 1, 3)
        let v = vProj(xs).reshaped([b, t, cfg.numHeads / cfg.kvRepeat, cfg.headDim]).transposed(0, 2, 1, 3)

        if let rope = rope {
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }

        var kFinal = k
        var vFinal = v
        if let cache = cache {
            (kFinal, vFinal) = cache.update(keys: k, values: v)
        }

        // Use the provided mask directly - mask generation happens in encode()
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: kFinal,
            values: vFinal,
            scale: scale,
            mask: mask
        )

        return oProj(out.transposed(0, 2, 1, 3).reshaped([b, t, -1]))
    }
}

/// MLP without gating (GELU)
public class EncoderMLP: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(_ cfg: EncoderTransformerConfig) {
        self._linear1.wrappedValue = Linear(cfg.dModel, cfg.dimFeedforward, bias: false)
        self._linear2.wrappedValue = Linear(cfg.dimFeedforward, cfg.dModel, bias: false)
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return linear2(geluApprox(linear1(xs)))
    }
}

/// Transformer layer for encoder
public class EncoderTransformerLayer: Module {
    let hasLayerScale: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: EncoderAttention
    @ModuleInfo(key: "gating") var gating: EncoderMLP
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "layer_scale_1") var layerScale1: EncoderLayerScale?
    @ModuleInfo(key: "layer_scale_2") var layerScale2: EncoderLayerScale?

    public init(_ cfg: EncoderTransformerConfig) {
        self.hasLayerScale = cfg.layerScale != nil
        self._selfAttn.wrappedValue = EncoderAttention(cfg)
        self._gating.wrappedValue = EncoderMLP(cfg)
        self._norm1.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)

        if let scale = cfg.layerScale {
            self._layerScale1.wrappedValue = EncoderLayerScale(dim: cfg.dModel, initialScale: scale)
            self._layerScale2.wrappedValue = EncoderLayerScale(dim: cfg.dModel, initialScale: scale)
        } else {
            self._layerScale1.wrappedValue = nil
            self._layerScale2.wrappedValue = nil
        }
    }

    public func callAsFunction(_ xs: MLXArray, cache: (any KVCache)?, mask: MLXArray?) -> MLXArray {
        var out = xs
        let n1 = norm1(out)
        let attnOut = selfAttn(n1, cache: cache, mask: mask)

        if let ls1 = layerScale1 {
            out = out + ls1(attnOut)
        } else {
            out = out + attnOut
        }

        let mlpOut = gating(norm2(out))
        if let ls2 = layerScale2 {
            out = out + ls2(mlpOut)
        } else {
            out = out + mlpOut
        }

        return out
    }
}

/// Transformer for encoder
public class EncoderTransformer: Module {
    let cfg: EncoderTransformerConfig

    @ModuleInfo(key: "layers") var layers: [EncoderTransformerLayer]

    public init(_ cfg: EncoderTransformerConfig) {
        self.cfg = cfg
        self._layers.wrappedValue = (0..<cfg.numLayers).map { _ in
            EncoderTransformerLayer(cfg)
        }
    }

    public func callAsFunction(_ xs: MLXArray, cache: [any KVCache]?, mask: MLXArray?) -> MLXArray {
        var out = xs
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            out = layer(out, cache: layerCache, mask: mask)
        }
        return out
    }

    public func makeCache() -> [KVCacheSimple] {
        return layers.map { _ in KVCacheSimple() }
    }
}

/// Projected transformer with input/output projections
public class EncoderProjectedTransformer: Module {
    let convLayout: Bool

    @ModuleInfo(key: "transformer") var transformer: EncoderTransformer
    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "output_projs") var outputProjs: [Linear?]

    public init(_ cfg: EncoderTransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout

        self._transformer.wrappedValue = EncoderTransformer(cfg)

        if inputDim == cfg.dModel {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Linear(inputDim, cfg.dModel, bias: false)
        }

        var projs: [Linear?] = []
        for outDim in outputDims {
            if outDim == cfg.dModel {
                projs.append(nil)
            } else {
                projs.append(Linear(cfg.dModel, outDim, bias: false))
            }
        }
        self._outputProjs.wrappedValue = projs
    }

    public func callAsFunction(_ xs: MLXArray, cache: [any KVCache]?, mask: MLXArray?) -> [MLXArray] {
        var out = xs
        if convLayout {
            out = out.transposed(0, 2, 1)  // NCL -> NLC
        }

        if let proj = inputProj {
            out = proj(out)
        }

        out = transformer(out, cache: cache, mask: mask)

        var results: [MLXArray] = []
        for proj in outputProjs {
            var result = out
            if let p = proj {
                result = p(result)
            }
            if convLayout {
                result = result.transposed(0, 2, 1)  // NLC -> NCL
            }
            results.append(result)
        }
        return results
    }

    public func makeCache() -> [KVCacheSimple] {
        return transformer.makeCache()
    }
}

// MARK: - Downsampling

/// Conv1d downsampling
public class EncoderConvDownsample1d: Module {
    @ModuleInfo(key: "conv") var conv: StreamableConv1d

    public init(stride: Int, dim: Int, causal: Bool) {
        self._conv.wrappedValue = StreamableConv1d(
            inChannels: dim,
            outChannels: dim,
            ksize: 2 * stride,
            stride: stride,
            dilation: 1,
            groups: 1,
            bias: false,
            causal: causal,
            padMode: "edge"
        )
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        return conv(xs)
    }
}

// MARK: - Vector Quantization (Encoder side)

/// Euclidean codebook for encoder
public class EncoderEuclideanCodebook: Module {
    let dim: Int
    let codebookSize: Int
    let epsilon: Float = 1e-5

    var embeddingSum: MLXArray
    var clusterUsage: MLXArray

    // Computed embeddings
    private var _embedding: MLXArray
    private var _c2: MLXArray

    // Public accessor for debugging
    public var embedding: MLXArray { _embedding }

    public init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.codebookSize = codebookSize

        self.embeddingSum = MLXArray.zeros([codebookSize, dim])
        self.clusterUsage = MLXArray.zeros([codebookSize])

        // Initialize computed values
        let usage = MLX.maximum(self.clusterUsage, MLXArray(epsilon)).expandedDimensions(axis: 1)
        self._embedding = self.embeddingSum / usage
        self._c2 = MLX.sum(MLX.pow(self._embedding, 2), axis: -1) / 2
    }

    /// Update computed embeddings from raw data
    public func updateInPlace() {
        let usage = MLX.maximum(clusterUsage, MLXArray(epsilon)).expandedDimensions(axis: 1)
        _embedding = embeddingSum / usage
        _c2 = MLX.sum(MLX.pow(_embedding, 2), axis: -1) / 2
        eval(_embedding, _c2)
    }

    /// Encode input to codebook indices
    public func encode(_ xs: MLXArray) -> MLXArray {
        let targetShape = Array(xs.shape.dropLast())
        let flat = xs.reshaped([-1, xs.dim(-1)])

        let xsF32 = flat.asType(.float32)
        let embedF32 = _embedding.asType(.float32)
        let c2F32 = _c2.asType(.float32)

        // transposed() on 2D array swaps dimensions: [2048, 256] -> [256, 2048]
        let dotProd = MLX.matmul(xsF32, embedF32.transposed())
        let distances = c2F32 - dotProd

        return argMin(distances, axis: -1).reshaped(targetShape)
    }

    /// Decode codebook indices to embeddings
    public func decode(_ xs: MLXArray) -> MLXArray {
        var targetShape = Array(xs.shape)
        targetShape.append(dim)
        return _embedding.take(xs.flattened(), axis: 0).reshaped(targetShape)
    }
}

/// Vector quantization layer for encoder
public class EncoderVectorQuantization: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear?
    @ModuleInfo(key: "project_out") var projectOut: Linear?
    @ModuleInfo(key: "codebook") var codebook: EncoderEuclideanCodebook

    public init(dim: Int, codebookSize: Int, codebookDim: Int?) {
        let cbDim = codebookDim ?? dim

        if dim == cbDim {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        } else {
            self._projectIn.wrappedValue = Linear(dim, cbDim)
            self._projectOut.wrappedValue = Linear(cbDim, dim)
        }

        self._codebook.wrappedValue = EncoderEuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
    }

    public func encode(_ xs: MLXArray) -> MLXArray {
        var out = xs.transposed(0, 2, 1)  // NCL -> NLC
        if let proj = projectIn {
            out = proj(out)
        }
        return codebook.encode(out)
    }

    public func decode(_ xs: MLXArray) -> MLXArray {
        var out = codebook.decode(xs)
        if let proj = projectOut {
            out = proj(out)
        }
        return out.transposed(0, 2, 1)  // NLC -> NCL
    }
}

/// Residual vector quantization for encoder
public class EncoderResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [EncoderVectorQuantization]

    public init(nq: Int, dim: Int, codebookSize: Int, codebookDim: Int?) {
        self._layers.wrappedValue = (0..<nq).map { _ in
            EncoderVectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    public func encode(_ xs: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = xs

        for layer in layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = residual.asType(.float32) - quantized.asType(.float32)
            residual = residual.asType(xs.dtype)
            codes.append(indices)
        }

        return MLX.stacked(codes, axis: 0)
    }

    public func decode(_ xs: MLXArray) -> MLXArray {
        var quantized = layers[0].decode(xs[0])
        for i in 1..<xs.dim(0) {
            quantized = quantized + layers[i].decode(xs[i])
        }
        return quantized
    }
}

/// Residual vector quantizer with projections for encoder
public class EncoderResidualVectorQuantizer: Module {
    @ModuleInfo(key: "input_proj") var inputProj: EncoderConv1dProj?
    @ModuleInfo(key: "output_proj") var outputProj: EncoderConv1dProj?
    @ModuleInfo(key: "vq") var vq: EncoderResidualVectorQuantization

    public init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int, forceProjection: Bool) {
        let inDim = inputDim ?? dim
        let outDim = outputDim ?? dim

        if inDim == dim && !forceProjection {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = EncoderConv1dProj(inChannels: inDim, outChannels: dim)
        }

        if outDim == dim && !forceProjection {
            self._outputProj.wrappedValue = nil
        } else {
            self._outputProj.wrappedValue = EncoderConv1dProj(inChannels: dim, outChannels: outDim)
        }

        self._vq.wrappedValue = EncoderResidualVectorQuantization(
            nq: nq,
            dim: dim,
            codebookSize: bins,
            codebookDim: nil
        )
    }

    public func encode(_ xs: MLXArray) -> MLXArray {
        var out = xs
        if let proj = inputProj {
            out = proj(out)
        }
        return vq.encode(out).transposed(1, 0, 2)  // [nq, batch, time] -> [batch, nq, time]
    }

    public func decode(_ xs: MLXArray) -> MLXArray {
        let codes = xs.transposed(1, 0, 2)  // [batch, nq, time] -> [nq, batch, time]
        var quantized = vq.decode(codes)
        if let proj = outputProj {
            quantized = proj(quantized)
        }
        return quantized
    }
}

/// 1x1 Conv1d projection for quantizer
public class EncoderConv1dProj: Module {
    var weight: MLXArray

    public init(inChannels: Int, outChannels: Int) {
        let scale = Float(1.0) / Float(inChannels)
        self.weight = MLXRandom.uniform(low: -scale, high: scale, [outChannels, 1, inChannels])
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        // xs: NCL
        let xNLC = xs.transposed(0, 2, 1)
        let y = MLX.conv1d(xNLC, weight, stride: 1, padding: 0)
        return y.transposed(0, 2, 1)
    }
}

/// Split residual vector quantizer for encoder
public class EncoderSplitResidualVectorQuantizer: Module {
    let nq: Int

    @ModuleInfo(key: "rvq_first") var rvqFirst: EncoderResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: EncoderResidualVectorQuantizer

    public init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int) {
        self.nq = nq

        self._rvqFirst.wrappedValue = EncoderResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: 1,
            bins: bins,
            forceProjection: true
        )

        self._rvqRest.wrappedValue = EncoderResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: nq - 1,
            bins: bins,
            forceProjection: true
        )
    }

    public func encode(_ xs: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(xs)
        if nq > 1 {
            let restCodes = rvqRest.encode(xs)
            codes = MLX.concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }

    public func decode(_ xs: MLXArray) -> MLXArray {
        var quantized = rvqFirst.decode(xs[0..., 0..<1, 0...])
        if nq > 1 {
            quantized = quantized + rvqRest.decode(xs[0..., 1..., 0...])
        }
        return quantized
    }
}

// MARK: - Main Encoder

/// Main speech tokenizer encoder
public class Qwen3TTSSpeechTokenizerEncoder: Module {
    let config: Qwen3TTSTokenizerEncoderConfig
    let validNumQuantizers: Int = 16  // Only first 16 quantizers are used for ICL

    @ModuleInfo(key: "encoder") var encoder: SeanetEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: EncoderProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: EncoderConvDownsample1d
    @ModuleInfo(key: "quantizer") var quantizer: EncoderSplitResidualVectorQuantizer

    var encoderCache: [KVCacheSimple]

    public init(_ config: Qwen3TTSTokenizerEncoderConfig) {
        self.config = config

        // Build SeanetConfig
        let seanetCfg = SeanetEncoderConfig(
            dimension: config.hiddenSize,
            channels: config.audioChannels,
            causal: config.useCausalConv,
            nfilters: config.numFilters,
            nresidualLayers: config.numResidualLayers,
            ratios: config.upsamplingRatios,
            ksize: config.kernelSize,
            residualKsize: config.residualKernelSize,
            lastKsize: config.lastKernelSize,
            dilationBase: config.dilationGrowthRate,
            padMode: "constant",
            trueSkip: !config.useConvShortcut,
            compress: config.compress
        )
        self._encoder.wrappedValue = SeanetEncoder(seanetCfg)

        // Build TransformerConfig
        let transformerCfg = EncoderTransformerConfig(
            dModel: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            causal: config.useCausalConv,
            layerScale: config.layerScaleInitialScale,
            context: config.slidingWindow,
            maxPeriod: Int(config.ropeTheta),
            maxSeqLen: config.maxPositionEmbeddings,
            kvRepeat: config.numAttentionHeads / config.numKeyValueHeads,
            dimFeedforward: config.intermediateSize,
            convLayout: true
        )
        self._encoderTransformer.wrappedValue = EncoderProjectedTransformer(
            transformerCfg,
            inputDim: config.hiddenSize,
            outputDims: [config.hiddenSize]
        )

        // Downsample: stride = encoder_frame_rate / frame_rate
        let encoderFrameRate = Float(config.samplingRate) / Float(config.upsamplingRatios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / config.frameRate)
        self._downsample.wrappedValue = EncoderConvDownsample1d(
            stride: downsampleStride,
            dim: config.hiddenSize,
            causal: config.useCausalConv
        )

        // Quantizer
        self._quantizer.wrappedValue = EncoderSplitResidualVectorQuantizer(
            dim: config.codebookDim,
            inputDim: config.hiddenSize,
            outputDim: config.hiddenSize,
            nq: config.numQuantizers,
            bins: config.codebookSize
        )

        self.encoderCache = []
    }

    /// Encode audio waveform to codes
    /// - Parameter audio: Audio waveform [batch, 1, samples]
    /// - Returns: Codes [batch, num_quantizers, time]
    public func encode(_ audio: MLXArray) -> MLXArray {
        // Reset cache
        encoderCache = encoderTransformer.makeCache()

        // Seanet encoder
        var xs = encoder(audio)

        // Create causal attention mask
        let seqLen = xs.dim(-1)
        let negInf = MLXArray(-Float.infinity)
        var mask = MLXArray.full([seqLen, seqLen], values: negInf)
        mask = MLX.triu(mask, k: 1)
        mask = mask.expandedDimensions(axes: [0, 1])

        // Transformer
        xs = encoderTransformer(xs, cache: encoderCache, mask: mask)[0]

        // Downsample
        xs = downsample(xs)

        // Quantize
        let codes = quantizer.encode(xs)

        // Return only first 16 quantizers
        return codes[0..., 0..<validNumQuantizers, 0...]
    }

    /// Initialize codebook embeddings from loaded weights
    public func initializeCodebooks() {
        // Update rvq_first codebooks
        for layer in quantizer.rvqFirst.vq.layers {
            layer.codebook.updateInPlace()
        }

        // Update rvq_rest codebooks
        for layer in quantizer.rvqRest.vq.layers {
            layer.codebook.updateInPlace()
        }
    }
}

// MARK: - Helper Activation

/// ELU activation function
private func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    return MLX.where(x .> 0, x, MLXArray(alpha) * (MLX.exp(x) - 1))
}

/// GELU approximate activation
private func geluApprox(_ x: MLXArray) -> MLXArray {
    return x * 0.5 * (1.0 + MLX.tanh(0.7978845608 * (x + 0.044715 * MLX.pow(x, 3))))
}
