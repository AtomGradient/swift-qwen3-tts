//
//  SpeechTokenizer.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  Qwen3 TTS Speech Tokenizer Decoder
//  Decodes codec tokens to audio waveform
//
//  Ported from Python mlx-audio implementation
//

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - Vector Quantizer

/// Codebook wrapper for weight loading
public class Codebook: Module {
    @ModuleInfo(key: "embed") var embed: Embedding

    public init(codebookSize: Int, dimension: Int) {
        self._embed.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dimension)
    }

    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        return embed(codes)
    }
}

/// Single codebook vector quantizer
public class VectorQuantizer: Module {
    let dimension: Int
    let codebookSize: Int

    /// Codebook embeddings [codebook_size, dimension]
    @ModuleInfo(key: "codebook") var codebook: Codebook

    public init(dimension: Int, codebookSize: Int) {
        self.dimension = dimension
        self.codebookSize = codebookSize
        self._codebook.wrappedValue = Codebook(codebookSize: codebookSize, dimension: dimension)
    }

    /// Decode codes to embeddings
    /// - Parameter codes: [batch, time]
    /// - Returns: [batch, dimension, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, time] -> lookup -> [batch, time, dimension]
        let embeddings = codebook(codes)
        // Transpose to [batch, dimension, time]
        return embeddings.transposed(0, 2, 1)
    }
}

// MARK: - Residual Vector Quantizer (Inner VQ)

/// Inner VQ that holds the codebook layers
public class ResidualVectorQuantizerVQ: Module {
    let numQuantizers: Int
    let dimension: Int
    let codebookSize: Int

    @ModuleInfo(key: "layers") var layers: [VectorQuantizer]

    public init(numQuantizers: Int, dimension: Int, codebookSize: Int) {
        self.numQuantizers = numQuantizers
        self.dimension = dimension
        self.codebookSize = codebookSize

        self._layers.wrappedValue = (0..<numQuantizers).map { _ in
            VectorQuantizer(dimension: dimension, codebookSize: codebookSize)
        }
    }

    /// Decode codes from all quantizers
    /// - Parameter codes: [batch, num_quantizers, time]
    /// - Returns: [batch, dimension, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        var quantized: MLXArray? = nil

        for i in 0..<numQuantizers {
            let layerCodes = codes[0..., i, 0...]  // [batch, time]
            let layerQuantized = layers[i].decode(layerCodes)

            if let existing = quantized {
                quantized = existing + layerQuantized
            } else {
                quantized = layerQuantized
            }
        }

        return quantized!
    }
}

// MARK: - Point-wise Conv1d Projection

/// 1x1 Conv1d projection for quantizer
public class Conv1dProjection: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(inChannels: Int, outChannels: Int) {
        // Weight shape: [out_channels, kernel_size=1, in_channels] in MLX
        self._weight.wrappedValue = MLXArray.zeros([outChannels, 1, inChannels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, in_channels, time]
        // Conv1d expects [batch, time, in_channels] in MLX, output is [batch, time, out_channels]
        let xT = x.transposed(0, 2, 1)  // [batch, time, in_channels]
        // Use conv1d: weight is [out, kernel, in]
        let out = conv1d(xT, weight, stride: 1, padding: 0, dilation: 1, groups: 1)
        // out: [batch, time, out_channels]
        return out.transposed(0, 2, 1)  // [batch, out_channels, time]
    }
}

// MARK: - Residual Vector Quantizer (with projections)

/// Residual VQ with input/output projections - matches weight file structure
public class ResidualVectorQuantizer: Module {
    let numQuantizers: Int
    let dimension: Int
    let codebookSize: Int
    let innerDimension: Int

    @ModuleInfo(key: "vq") var vq: ResidualVectorQuantizerVQ
    @ModuleInfo(key: "input_proj") var inputProj: Conv1dProjection
    @ModuleInfo(key: "output_proj") var outputProj: Conv1dProjection

    public init(numQuantizers: Int, dimension: Int, codebookSize: Int, innerDimension: Int? = nil) {
        self.numQuantizers = numQuantizers
        self.dimension = dimension
        self.codebookSize = codebookSize
        // Inner dimension is typically dimension / 2 (512 -> 256)
        self.innerDimension = innerDimension ?? (dimension / 2)

        self._vq.wrappedValue = ResidualVectorQuantizerVQ(
            numQuantizers: numQuantizers,
            dimension: self.innerDimension,
            codebookSize: codebookSize
        )

        // Create projections: input compresses, output expands
        self._inputProj.wrappedValue = Conv1dProjection(
            inChannels: dimension,
            outChannels: self.innerDimension
        )
        self._outputProj.wrappedValue = Conv1dProjection(
            inChannels: self.innerDimension,
            outChannels: dimension
        )
    }

    /// Decode codes from all quantizers
    /// - Parameter codes: [batch, num_quantizers, time]
    /// - Returns: [batch, dimension, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        // VQ decode: get embeddings from codebooks
        var quantized = vq.decode(codes)

        // Apply output projection to expand back to full dimension
        quantized = outputProj(quantized)

        return quantized
    }
}

// MARK: - Split Residual Vector Quantizer

/// Split RVQ: Semantic (1 quantizer) + Acoustic (15 quantizers)
public class SplitResidualVectorQuantizer: Module {
    let numSemanticQuantizers: Int
    let numAcousticQuantizers: Int
    let dimension: Int

    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    public init(
        dimension: Int,
        numQuantizers: Int,
        numSemanticQuantizers: Int,
        codebookSize: Int,
        semanticCodebookSize: Int? = nil
    ) {
        self.numSemanticQuantizers = numSemanticQuantizers
        self.numAcousticQuantizers = numQuantizers - numSemanticQuantizers
        self.dimension = dimension

        let semanticSize = semanticCodebookSize ?? codebookSize

        // Semantic quantizer (first 1)
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: numSemanticQuantizers,
            dimension: dimension,
            codebookSize: semanticSize
        )

        // Acoustic quantizers (remaining 15)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: numQuantizers - numSemanticQuantizers,
            dimension: dimension,
            codebookSize: codebookSize
        )
    }

    /// Decode codes to audio features
    /// - Parameter codes: [batch, num_quantizers, time]
    /// - Returns: [batch, output_dimension, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        // Decode semantic codes (first quantizer)
        let semanticCodes = codes[0..., ..<numSemanticQuantizers, 0...]
        var quantized = rvqFirst.decode(semanticCodes)

        // Decode acoustic codes (remaining quantizers) if present
        if codes.dim(1) > numSemanticQuantizers {
            let acousticCodes = codes[0..., numSemanticQuantizers..., 0...]
            quantized = quantized + rvqRest.decode(acousticCodes)
        }

        return quantized
    }
}

// MARK: - SnakeBeta Activation

/// Snake activation: x + (1/beta) * sin^2(x * alpha)
public class SnakeBeta: Module {
    let channels: Int
    @ModuleInfo var alpha: MLXArray
    @ModuleInfo var beta: MLXArray

    let eps: Float = 1e-9

    public init(channels: Int) {
        self.channels = channels
        // Initialize as log values (will be exp'd)
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let alphaExp = exp(alpha).reshaped([1, channels, 1])
        let betaExp = exp(beta).reshaped([1, channels, 1])

        let sinTerm = sin(x * alphaExp)
        return x + (1.0 / (betaExp + eps)) * (sinTerm * sinTerm)
    }
}

// MARK: - Causal Conv1d

/// Causal 1D convolution with left padding
public class CausalConv1d: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let dilation: Int
    let groups: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.groups = groups

        // No padding in conv, we'll do causal padding manually
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation,
            groups: groups
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCT format)
        // MLX Conv1d expects [batch, time, channels] (NTC format)
        let xT = x.transposed(0, 2, 1)  // [batch, time, channels]

        let padding = (kernelSize - 1) * dilation

        // Pad on the left (causal) - now padding time dimension which is axis 1
        let padded = MLX.padded(xT, widths: [[0, 0], [padding, 0], [0, 0]])

        let out = conv(padded)  // [batch, time, out_channels]
        return out.transposed(0, 2, 1)  // [batch, out_channels, time]
    }
}

// MARK: - Causal Transpose Conv1d

/// Causal transpose 1D convolution for upsampling
public class CausalTransposeConv1d: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int

    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride

        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCT format)
        // MLX ConvTransposed1d expects [batch, time, channels] (NTC format)
        let xT = x.transposed(0, 2, 1)  // [batch, time, channels]

        let out = conv(xT)  // [batch, out_time, out_channels]

        // Trim to causal output (remove right padding)
        let trimAmount = kernelSize - stride
        if trimAmount > 0 {
            let trimmed = out[0..., ..<(-trimAmount), 0...]  // [batch, time, channels]
            return trimmed.transposed(0, 2, 1)  // [batch, channels, time]
        }
        return out.transposed(0, 2, 1)  // [batch, channels, time]
    }
}

// MARK: - ConvNeXt Block

/// ConvNeXt-style block with depthwise conv
public class ConvNeXtBlock: Module {
    let dim: Int

    @ModuleInfo(key: "dwconv") var dwconv: CausalConv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ModuleInfo var gamma: MLXArray

    public init(dim: Int) {
        self.dim = dim

        // Depthwise convolution
        self._dwconv.wrappedValue = CausalConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 7,
            groups: dim
        )

        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, dim * 4)
        self._pwconv2.wrappedValue = Linear(dim * 4, dim)
        self.gamma = MLXArray.ones([dim]) * 1e-6
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let residual = x

        var h = dwconv(x)

        // Transpose for LayerNorm: [batch, channels, time] -> [batch, time, channels]
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        h = gamma * h
        h = h.transposed(0, 2, 1)

        return residual + h
    }
}

// MARK: - Decoder Residual Unit

/// Residual unit with dilated convolutions
/// Weight keys: act1, conv1, act2, conv2
public class DecoderResidualUnit: Module {
    @ModuleInfo(key: "act1") var snake1: SnakeBeta
    @ModuleInfo(key: "conv1") var conv1: CausalConv1d
    @ModuleInfo(key: "act2") var snake2: SnakeBeta
    @ModuleInfo(key: "conv2") var conv2: CausalConv1d

    public init(dim: Int, dilation: Int) {
        self._snake1.wrappedValue = SnakeBeta(channels: dim)
        self._conv1.wrappedValue = CausalConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 7,
            dilation: dilation
        )
        self._snake2.wrappedValue = SnakeBeta(channels: dim)
        self._conv2.wrappedValue = CausalConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return residual + h
    }
}

// MARK: - Decoder Block (Upsampling)

/// Decoder block with upsampling and residual units
/// Weight keys use "block.X" format which needs remapping in sanitize function
public class DecoderBlock: Module {
    let inDim: Int
    let outDim: Int
    let upsampleRate: Int

    @ModuleInfo(key: "snake") var snake: SnakeBeta
    @ModuleInfo(key: "upsample") var upsample: CausalTransposeConv1d
    @ModuleInfo(key: "res1") var res1: DecoderResidualUnit
    @ModuleInfo(key: "res2") var res2: DecoderResidualUnit
    @ModuleInfo(key: "res3") var res3: DecoderResidualUnit

    public init(_ config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        let decoderDim = config.decoderDim
        self.inDim = decoderDim / (1 << layerIdx)
        self.outDim = decoderDim / (1 << (layerIdx + 1))
        self.upsampleRate = config.upsampleRates[layerIdx]

        self._snake.wrappedValue = SnakeBeta(channels: inDim)
        self._upsample.wrappedValue = CausalTransposeConv1d(
            inChannels: inDim,
            outChannels: outDim,
            kernelSize: upsampleRate * 2,
            stride: upsampleRate
        )
        self._res1.wrappedValue = DecoderResidualUnit(dim: outDim, dilation: 1)
        self._res2.wrappedValue = DecoderResidualUnit(dim: outDim, dilation: 3)
        self._res3.wrappedValue = DecoderResidualUnit(dim: outDim, dilation: 9)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = upsample(h)
        h = res1(h)
        h = res2(h)
        h = res3(h)
        return h
    }
}

// MARK: - Decoder Transformer Attention

public class DecoderTransformerAttention: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(_ config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Layer Scale

/// Layer scale for residual connections (ConvNeXt style)
public class LayerScale: Module {
    @ModuleInfo(key: "scale") var scale: MLXArray

    public init(dimensions: Int, initialScale: Float = 0.01) {
        self._scale.wrappedValue = MLXArray.ones([dimensions]) * initialScale
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * scale
    }
}

// MARK: - Decoder MLP

/// MLP for decoder transformer
public class DecoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Transformer Layer

public class DecoderTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DecoderTransformerAttention
    @ModuleInfo(key: "mlp") var mlp: DecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "self_attn_layer_scale") var selfAttnLayerScale: LayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpLayerScale: LayerScale

    public init(_ config: Qwen3TTSTokenizerDecoderConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._selfAttn.wrappedValue = DecoderTransformerAttention(config)
        self._mlp.wrappedValue = DecoderMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        self._selfAttnLayerScale.wrappedValue = LayerScale(dimensions: hiddenSize, initialScale: config.layerScaleInitialScale)
        self._mlpLayerScale.wrappedValue = LayerScale(dimensions: hiddenSize, initialScale: config.layerScaleInitialScale)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var residual = x
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask)
        h = selfAttnLayerScale(h)
        h = residual + h

        residual = h
        h = postAttentionLayernorm(h)
        h = mlp(h)
        h = mlpLayerScale(h)
        h = residual + h

        return h
    }
}

// MARK: - Decoder Transformer

public class DecoderTransformer: Module {
    let hiddenSize: Int
    let latentDim: Int

    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    let layers: [DecoderTransformerLayer]
    let norm: RMSNorm

    public init(_ config: Qwen3TTSTokenizerDecoderConfig) {
        self.hiddenSize = config.hiddenSize
        self.latentDim = config.latentDim

        // Input/output projections
        self._inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize, bias: true)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim, bias: true)

        self.layers = (0..<config.numHiddenLayers).map { _ in
            DecoderTransformerLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Input projection: [batch, time, latent] -> [batch, time, hidden]
        var h = inputProj(x)

        // Transformer layers
        for layer in layers {
            h = layer(h, mask: mask)
        }

        // Normalize and output projection
        h = norm(h)
        h = outputProj(h)

        return h
    }
}

// MARK: - Main Decoder

/// Main decoder module
/// Python structure: self.decoder = [DecoderInitialConv, DecoderBlock, ..., DecoderOutputSnake, DecoderOutputConv]
/// Weight file uses numeric keys (decoder.decoder.0, decoder.decoder.1, etc.)
/// We use named keys here and remap in sanitizeSpeechTokenizerWeights
public class MainDecoder: Module {
    @ModuleInfo(key: "initConv") var initConv: CausalConv1d
    @ModuleInfo(key: "block0") var block0: DecoderBlock
    @ModuleInfo(key: "block1") var block1: DecoderBlock
    @ModuleInfo(key: "block2") var block2: DecoderBlock
    @ModuleInfo(key: "block3") var block3: DecoderBlock
    @ModuleInfo(key: "outSnake") var outSnake: SnakeBeta
    @ModuleInfo(key: "outConv") var outConv: CausalConv1d

    public init(_ config: Qwen3TTSTokenizerDecoderConfig) {
        self._initConv.wrappedValue = CausalConv1d(
            inChannels: config.latentDim,
            outChannels: config.decoderDim,
            kernelSize: 7
        )
        self._block0.wrappedValue = DecoderBlock(config, layerIdx: 0)  // 8x
        self._block1.wrappedValue = DecoderBlock(config, layerIdx: 1)  // 5x
        self._block2.wrappedValue = DecoderBlock(config, layerIdx: 2)  // 4x
        self._block3.wrappedValue = DecoderBlock(config, layerIdx: 3)  // 3x

        let outputDim = config.decoderDim / (1 << 4)  // After 4 blocks
        self._outSnake.wrappedValue = SnakeBeta(channels: outputDim)
        self._outConv.wrappedValue = CausalConv1d(
            inChannels: outputDim,
            outChannels: 1,  // Mono audio
            kernelSize: 7
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var wav = initConv(x)
        wav = block0(wav)
        wav = block1(wav)
        wav = block2(wav)
        wav = block3(wav)
        wav = outSnake(wav)
        wav = outConv(wav)
        return wav
    }
}

// MARK: - Speech Tokenizer Decoder

/// Main decoder: codes -> audio waveform
public class Qwen3TTSSpeechTokenizerDecoder: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let totalUpsample: Int

    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer
    @ModuleInfo(key: "pre_conv") var preConv: CausalConv1d
    @ModuleInfo(key: "pre_transformer") var preTransformer: DecoderTransformer

    /// Upsampling modules (2x * 2x = 4x)
    @ModuleInfo(key: "upsample") var upsample: [[Module]]

    /// Main decoder blocks (480x upsampling)
    @ModuleInfo(key: "decoder") var mainDecoder: MainDecoder

    public init(_ config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self.totalUpsample = config.totalUpsample

        // Split RVQ quantizer
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dimension: config.codebookDim,  // 512
            numQuantizers: config.numQuantizers,
            numSemanticQuantizers: config.numSemanticQuantizers,
            codebookSize: config.codebookSize,
            semanticCodebookSize: config.semanticCodebookSize
        )

        // Pre-conv
        self._preConv.wrappedValue = CausalConv1d(
            inChannels: config.codebookDim,
            outChannels: config.latentDim,
            kernelSize: 3
        )

        // Pre-transformer
        self._preTransformer.wrappedValue = DecoderTransformer(config)

        // Upsampling (2x * 2x = 4x)
        var upsampleLayers: [[Module]] = []
        for ratio in config.upsamplingRatios {
            let transposeConv = CausalTransposeConv1d(
                inChannels: config.latentDim,
                outChannels: config.latentDim,
                kernelSize: ratio,
                stride: ratio
            )
            let convNext = ConvNeXtBlock(dim: config.latentDim)
            upsampleLayers.append([transposeConv, convNext])
        }
        self._upsample.wrappedValue = upsampleLayers

        // Main decoder
        self._mainDecoder.wrappedValue = MainDecoder(config)
    }

    /// Decode codes to audio
    /// - Parameter codes: [batch, num_quantizers, time]
    /// - Returns: [batch, 1, samples]
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        // 1. Dequantize
        var hidden = quantizer.decode(codes)  // [batch, codebook_dim, time]

        // 2. Pre-conv
        hidden = preConv(hidden)  // [batch, latent_dim, time]

        // 3. Pre-transformer (need to transpose)
        hidden = hidden.transposed(0, 2, 1)  // [batch, time, latent_dim]
        hidden = preTransformer(hidden)
        hidden = hidden.transposed(0, 2, 1)  // [batch, latent_dim, time]

        // 4. Upsampling (4x)
        for layers in upsample {
            for layer in layers {
                if let conv = layer as? CausalTransposeConv1d {
                    hidden = conv(hidden)
                } else if let block = layer as? ConvNeXtBlock {
                    hidden = block(hidden)
                }
            }
        }

        // 5. Main decoder (480x)
        var wav = mainDecoder(hidden)

        // 6. Clip to [-1, 1]
        wav = MLX.clip(wav, min: -1.0, max: 1.0)

        return wav
    }
}

// MARK: - Speech Tokenizer

/// Main speech tokenizer class
public class Qwen3TTSSpeechTokenizer: Module {
    let decodeUpsampleRate: Int  // 1920
    let encodeDownsampleRate: Int  // 1920

    @ModuleInfo(key: "decoder") var decoder: Qwen3TTSSpeechTokenizerDecoder
    @ModuleInfo(key: "encoder") var encoder: Qwen3TTSSpeechTokenizerEncoder?

    public init(_ config: Qwen3TTSTokenizerConfig) {
        self.decodeUpsampleRate = config.decodeUpsampleRate
        self.encodeDownsampleRate = config.encodeDownsampleRate

        if let decoderConfig = config.decoderConfig {
            self._decoder.wrappedValue = Qwen3TTSSpeechTokenizerDecoder(decoderConfig)
        } else {
            fatalError("Decoder config is required")
        }

        // Initialize encoder if config is available
        if let encoderConfig = config.encoderConfig {
            self._encoder.wrappedValue = Qwen3TTSSpeechTokenizerEncoder(encoderConfig)
        } else {
            self._encoder.wrappedValue = nil
        }
    }

    /// Check if encoder is available for voice cloning
    public var hasEncoder: Bool {
        return encoder != nil
    }

    /// Decode codes to audio
    /// - Parameter audioCodes: [batch, seq_len, num_quantizers]
    /// - Returns: (audio [batch, samples], audio_lengths [batch])
    public func decode(_ audioCodes: MLXArray) -> (audio: MLXArray, audioLengths: MLXArray) {
        // Transpose: [batch, seq_len, 16] -> [batch, 16, seq_len]
        let codes = audioCodes.transposed(0, 2, 1)

        // Decode
        let wav = decoder(codes).squeezed(axis: 1)  // [batch, samples]

        // Calculate valid lengths based on non-padding codes
        let firstCodebook = audioCodes[0..., 0..., 0]  // [batch, seq_len]
        let validTokens = MLX.sum(firstCodebook .> 0, axis: 1)
        let audioLengths = validTokens * decodeUpsampleRate

        return (wav, audioLengths)
    }

    /// Encode audio to codes
    /// - Parameter audio: Audio waveform [batch, 1, samples]
    /// - Returns: Codes [batch, num_quantizers, time]
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        guard let encoder = encoder else {
            throw Qwen3TTSSpeechTokenizerError.encoderNotAvailable
        }
        return encoder.encode(audio)
    }

    /// Initialize encoder codebooks after loading weights
    public func initializeEncoderCodebooks() {
        encoder?.initializeCodebooks()
    }
}

// MARK: - Errors

public enum Qwen3TTSSpeechTokenizerError: Error, LocalizedError {
    case encoderNotAvailable

    public var errorDescription: String? {
        switch self {
        case .encoderNotAvailable:
            return "Speech tokenizer encoder is not yet implemented in Swift. Voice cloning (ICL mode) requires the encoder to convert reference audio to codes."
        }
    }
}
