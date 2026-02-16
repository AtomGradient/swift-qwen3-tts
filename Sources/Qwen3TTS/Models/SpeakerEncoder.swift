//
//  SpeakerEncoder.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//  ECAPA-TDNN Speaker Encoder for Qwen3-TTS Voice Cloning
//  Ported from Python mlx-audio implementation
//

import Foundation
import MLX
import MLXNN

// MARK: - Reflect Padding

/// Reverse array along a given axis
private func reverseAxis(_ x: MLXArray, axis: Int) -> MLXArray {
    // Use negative strides to reverse - get indices in reverse order
    let dimSize = x.dim(axis)
    let indices = MLXArray(Array((0..<dimSize).reversed()).map { Int32($0) })
    return x.take(indices, axis: axis)
}

/// Apply reflect padding to the time dimension (axis=1) in NLC format
func reflectPad1d(_ x: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return x }

    // Reflect: mirror without repeating the boundary element
    // left = x[:, 1:pad+1, :][:, ::-1, :]
    let leftSlice = x[0..., 1..<(pad + 1), 0...]
    let left = reverseAxis(leftSlice, axis: 1)

    // right = x[:, -(pad+1):-1, :][:, ::-1, :]
    let timeLen = x.dim(1)
    let rightSlice = x[0..., (timeLen - pad - 1)..<(timeLen - 1), 0...]
    let right = reverseAxis(rightSlice, axis: 1)

    return MLX.concatenated([left, x, right], axis: 1)
}

// MARK: - TimeDelayNetBlock

/// TDNN block with 1D convolution, reflect padding, and ReLU activation
public class TimeDelayNetBlock: Module {
    let pad: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
        // Compute "same" padding amount
        self.pad = (kernelSize - 1) * dilation / 2
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        var out = x.transposed(0, 2, 1)  // NCL -> NLC
        out = reflectPad1d(out, pad: pad)
        out = conv(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL
        return relu(out)
    }
}

// MARK: - Res2NetBlock

/// Res2Net block for multi-scale feature extraction
public class Res2NetBlock: Module {
    let scale: Int
    @ModuleInfo(key: "blocks") var blocks: [TimeDelayNetBlock]

    public init(inChannels: Int, outChannels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        self.scale = scale
        let inChannel = inChannels / scale
        let hiddenChannel = outChannels / scale

        var blockList: [TimeDelayNetBlock] = []
        for _ in 0..<(scale - 1) {
            blockList.append(TimeDelayNetBlock(
                inChannels: inChannel,
                outChannels: hiddenChannel,
                kernelSize: kernelSize,
                dilation: dilation
            ))
        }
        self._blocks.wrappedValue = blockList
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let channelSize = x.dim(1) / scale
        var outputs: [MLXArray] = []
        var outputPart: MLXArray? = nil

        for i in 0..<scale {
            let chunk = x[0..., (i * channelSize)..<((i + 1) * channelSize), 0...]

            if i == 0 {
                outputPart = chunk
            } else if i == 1 {
                outputPart = blocks[i - 1](chunk)
            } else {
                outputPart = blocks[i - 1](chunk + outputPart!)
            }
            outputs.append(outputPart!)
        }

        return MLX.concatenated(outputs, axis: 1)
    }
}

// MARK: - SqueezeExcitationBlock

/// Squeeze-and-excitation block for channel attention
public class SqueezeExcitationBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    public init(inChannels: Int, seChannels: Int, outChannels: Int) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: seChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: seChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        // Global average pooling
        let xMean = x.mean(axis: 2, keepDims: true)  // [batch, channels, 1]

        // SE path - transpose for MLX Conv1d (NLC format)
        var se = xMean.transposed(0, 2, 1)  // [batch, 1, channels]
        se = relu(conv1(se))
        se = sigmoid(conv2(se))
        se = se.transposed(0, 2, 1)  // [batch, channels, 1]

        return x * se
    }
}

// MARK: - SqueezeExcitationRes2NetBlock

/// TDNN-Res2Net-TDNN-SE block used in ECAPA-TDNN
public class SqueezeExcitationRes2NetBlock: Module {
    let outChannels: Int
    @ModuleInfo(key: "tdnn1") var tdnn1: TimeDelayNetBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo(key: "tdnn2") var tdnn2: TimeDelayNetBlock
    @ModuleInfo(key: "se_block") var seBlock: SqueezeExcitationBlock

    public init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        self.outChannels = outChannels

        self._tdnn1.wrappedValue = TimeDelayNetBlock(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._res2netBlock.wrappedValue = Res2NetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            scale: res2netScale,
            kernelSize: kernelSize,
            dilation: dilation
        )
        self._tdnn2.wrappedValue = TimeDelayNetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._seBlock.wrappedValue = SqueezeExcitationBlock(
            inChannels: outChannels,
            seChannels: seChannels,
            outChannels: outChannels
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = tdnn1(x)
        out = res2netBlock(out)
        out = tdnn2(out)
        out = seBlock(out)
        return out + residual
    }
}

// MARK: - AttentiveStatisticsPooling

/// Attentive statistics pooling layer
public class AttentiveStatisticsPooling: Module {
    let eps: Float = 1e-12
    @ModuleInfo(key: "tdnn") var tdnn: TimeDelayNetBlock
    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(channels: Int, attentionChannels: Int = 128) {
        self._tdnn.wrappedValue = TimeDelayNetBlock(
            inChannels: channels * 3,
            outChannels: attentionChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._conv.wrappedValue = Conv1d(
            inputChannels: attentionChannels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let (batch, channels, seqLength) = (x.dim(0), x.dim(1), x.dim(2))

        // Compute mean and std
        let mean = x.mean(axis: 2, keepDims: true)
        let variance = x.variance(axis: 2, keepDims: true)
        let std = MLX.sqrt(variance + eps)

        // Expand to match sequence length
        let meanExpanded = MLX.broadcast(mean, to: [batch, channels, seqLength])
        let stdExpanded = MLX.broadcast(std, to: [batch, channels, seqLength])

        // Concatenate features
        var attention = MLX.concatenated([x, meanExpanded, stdExpanded], axis: 1)

        // Apply attention
        attention = tdnn(attention)
        attention = tanh(attention)

        // Conv expects NLC format
        attention = attention.transposed(0, 2, 1)  // NCL -> NLC
        attention = conv(attention)
        attention = attention.transposed(0, 2, 1)  // NLC -> NCL
        attention = softmax(attention, axis: 2)

        // Compute weighted mean and std
        let weightedMean = (attention * x).sum(axis: 2, keepDims: true)
        let weightedVar = (attention * MLX.pow(x - weightedMean, 2)).sum(axis: 2, keepDims: true)
        let weightedStd = MLX.sqrt(MLX.clip(weightedVar, min: eps, max: Float.greatestFiniteMagnitude))

        // Concatenate mean and std
        let pooled = MLX.concatenated([weightedMean, weightedStd], axis: 1)
        return pooled
    }
}

// MARK: - Qwen3TTSSpeakerEncoder

/// ECAPA-TDNN speaker encoder for Qwen3-TTS voice cloning
public class Qwen3TTSSpeakerEncoder: Module {
    let config: Qwen3TTSSpeakerEncoderConfig
    let channels: [Int]

    // Initial TDNN layer
    @ModuleInfo(key: "blocks.0") var initialTdnn: TimeDelayNetBlock

    // SE-Res2Net layers
    @ModuleInfo(key: "blocks.1") var seRes2Net1: SqueezeExcitationRes2NetBlock
    @ModuleInfo(key: "blocks.2") var seRes2Net2: SqueezeExcitationRes2NetBlock
    @ModuleInfo(key: "blocks.3") var seRes2Net3: SqueezeExcitationRes2NetBlock

    // Multi-layer feature aggregation
    @ModuleInfo(key: "mfa") var mfa: TimeDelayNetBlock

    // Attentive Statistical Pooling
    @ModuleInfo(key: "asp") var asp: AttentiveStatisticsPooling

    // Final linear transformation
    @ModuleInfo(key: "fc") var fc: Conv1d

    public init(_ config: Qwen3TTSSpeakerEncoderConfig) {
        self.config = config
        self.channels = config.encChannels

        // Initial TDNN layer
        self._initialTdnn.wrappedValue = TimeDelayNetBlock(
            inChannels: config.melDim,
            outChannels: config.encChannels[0],
            kernelSize: config.encKernelSizes[0],
            dilation: config.encDilations[0]
        )

        // SE-Res2Net layers (indices 1, 2, 3)
        self._seRes2Net1.wrappedValue = SqueezeExcitationRes2NetBlock(
            inChannels: config.encChannels[0],
            outChannels: config.encChannels[1],
            res2netScale: config.encRes2netScale,
            seChannels: config.encSeChannels,
            kernelSize: config.encKernelSizes[1],
            dilation: config.encDilations[1]
        )
        self._seRes2Net2.wrappedValue = SqueezeExcitationRes2NetBlock(
            inChannels: config.encChannels[1],
            outChannels: config.encChannels[2],
            res2netScale: config.encRes2netScale,
            seChannels: config.encSeChannels,
            kernelSize: config.encKernelSizes[2],
            dilation: config.encDilations[2]
        )
        self._seRes2Net3.wrappedValue = SqueezeExcitationRes2NetBlock(
            inChannels: config.encChannels[2],
            outChannels: config.encChannels[3],
            res2netScale: config.encRes2netScale,
            seChannels: config.encSeChannels,
            kernelSize: config.encKernelSizes[3],
            dilation: config.encDilations[3]
        )

        // Multi-layer feature aggregation
        // Input: concatenation of 3 SE-Res2Net outputs = encChannels[1] + encChannels[2] + encChannels[3]
        let mfaInputChannels = config.encChannels[1] + config.encChannels[2] + config.encChannels[3]
        self._mfa.wrappedValue = TimeDelayNetBlock(
            inChannels: mfaInputChannels,
            outChannels: config.encChannels[4],
            kernelSize: config.encKernelSizes[4],
            dilation: config.encDilations[4]
        )

        // Attentive Statistical Pooling
        self._asp.wrappedValue = AttentiveStatisticsPooling(
            channels: config.encChannels[4],
            attentionChannels: config.encAttentionChannels
        )

        // Final linear transformation
        // ASP outputs channels * 2 (mean + std)
        self._fc.wrappedValue = Conv1d(
            inputChannels: config.encChannels[4] * 2,
            outputChannels: config.encDim,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: Mel spectrogram [batch, time, mel_dim]

        // Transpose to [batch, channels, time]
        var out = x.transposed(0, 2, 1)

        // Initial TDNN
        let h0 = initialTdnn(out)

        // SE-Res2Net layers
        let h1 = seRes2Net1(h0)
        let h2 = seRes2Net2(h1)
        let h3 = seRes2Net3(h2)

        // Multi-layer feature aggregation (concatenate SE-Res2Net outputs)
        out = MLX.concatenated([h1, h2, h3], axis: 1)
        out = mfa(out)

        // Attentive Statistical Pooling
        out = asp(out)

        // Final linear transformation - Conv expects NLC format
        out = out.transposed(0, 2, 1)  // NCL -> NLC
        out = fc(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL

        // Squeeze time dimension: [batch, enc_dim, 1] -> [batch, enc_dim]
        out = out.squeezed(axis: -1)

        return out
    }
}

// MARK: - Mel Spectrogram

/// Compute mel spectrogram from audio waveform
/// - Parameters:
///   - audio: Audio waveform [samples] or [batch, samples]
///   - nFft: FFT size
///   - numMels: Number of mel bins
///   - sampleRate: Audio sample rate
///   - hopSize: Hop size in samples
///   - winSize: Window size in samples
///   - fMin: Minimum frequency
///   - fMax: Maximum frequency
/// - Returns: Mel spectrogram [batch, time, mels]
public func melSpectrogram(
    _ audio: MLXArray,
    nFft: Int = 1024,
    numMels: Int = 128,
    sampleRate: Int = 24000,
    hopSize: Int = 256,
    winSize: Int = 1024,
    fMin: Float = 0,
    fMax: Float = 12000
) -> MLXArray {
    // Ensure 2D input [batch, samples]
    var x = audio
    if x.ndim == 1 {
        x = x.expandedDimensions(axis: 0)
    }

    // Create Hann window
    let window = hannWindow(winSize)

    // Pad audio for STFT using constant padding (reflect padding not available in MLX Swift)
    let padAmount = nFft / 2
    let padded = MLX.padded(x, widths: [[0, 0], [padAmount, padAmount]], mode: .constant, value: MLXArray(Float(0)))

    // Compute STFT
    let stft = computeSTFT(padded, nFft: nFft, hopSize: hopSize, window: window)

    // Compute power spectrogram
    let power = MLX.pow(MLX.abs(stft), 2)

    // Create mel filterbank
    let melFilter = melFilterbank(
        nFft: nFft,
        numMels: numMels,
        sampleRate: sampleRate,
        fMin: fMin,
        fMax: fMax
    )

    // Apply mel filterbank: [batch, freq, time] @ [freq, mels] -> [batch, mels, time]
    let melSpec = MLX.matmul(power.transposed(0, 2, 1), melFilter).transposed(0, 2, 1)

    // Convert to log scale
    let logMelSpec = MLX.log(MLX.maximum(melSpec, MLXArray(1e-10)))

    // Transpose to [batch, time, mels]
    return logMelSpec.transposed(0, 2, 1)
}

/// Create Hann window
private func hannWindow(_ size: Int) -> MLXArray {
    let n = MLXArray(Array(0..<size).map { Float($0) })
    return 0.5 * (1 - MLX.cos(2 * Float.pi * n / Float(size - 1)))
}

/// Compute Short-Time Fourier Transform
private func computeSTFT(_ x: MLXArray, nFft: Int, hopSize: Int, window: MLXArray) -> MLXArray {
    let (batch, samples) = (x.dim(0), x.dim(1))

    // Calculate number of frames
    let numFrames = (samples - nFft) / hopSize + 1

    // Extract frames
    var frames: [MLXArray] = []
    for i in 0..<numFrames {
        let start = i * hopSize
        let frame = x[0..., start..<(start + nFft)] * window
        frames.append(frame)
    }

    // Stack frames: [batch, numFrames, nFft]
    let stackedFrames = MLX.stacked(frames, axis: 1)

    // Compute FFT using MLXFFT
    // Note: MLX Swift uses fft() not rfft(), we need to take only positive frequencies
    let fullFft = MLXFFT.fft(stackedFrames, axis: -1)
    let numFreqs = nFft / 2 + 1
    let rfft = fullFft[0..., 0..., 0..<numFreqs]

    // Transpose to [batch, freq, time]
    return rfft.transposed(0, 2, 1)
}

/// Create mel filterbank matrix
private func melFilterbank(
    nFft: Int,
    numMels: Int,
    sampleRate: Int,
    fMin: Float,
    fMax: Float
) -> MLXArray {
    // Convert Hz to Mel
    func hzToMel(_ hz: Float) -> Float {
        return 2595.0 * log10(1.0 + hz / 700.0)
    }

    // Convert Mel to Hz
    func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    let melMin = hzToMel(fMin)
    let melMax = hzToMel(fMax)

    // Create mel points
    var melPoints: [Float] = []
    for i in 0...(numMels + 1) {
        let mel = melMin + Float(i) * (melMax - melMin) / Float(numMels + 1)
        melPoints.append(melToHz(mel))
    }

    // Convert to FFT bin indices
    let fftBins = melPoints.map { Int(floor(Float(nFft + 1) * $0 / Float(sampleRate))) }

    // Create filterbank
    let numFreqs = nFft / 2 + 1
    var filterbank: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numMels), count: numFreqs)

    for m in 0..<numMels {
        let left = fftBins[m]
        let center = fftBins[m + 1]
        let right = fftBins[m + 2]

        // Left slope
        for k in left..<center {
            if k < numFreqs && center > left {
                filterbank[k][m] = Float(k - left) / Float(center - left)
            }
        }

        // Right slope
        for k in center..<right {
            if k < numFreqs && right > center {
                filterbank[k][m] = Float(right - k) / Float(right - center)
            }
        }
    }

    // Flatten and create MLXArray
    let flatData = filterbank.flatMap { $0 }
    return MLXArray(flatData).reshaped([numFreqs, numMels])
}
