//
//  AudioUtils.swift
//  AtomGradient
//
//  Copyright (c) AtomGradient. All rights reserved.
//
//

import AVFoundation
import Foundation
import MLX

/// Load audio from a file and return the sample rate and audio data.
public func loadAudioArray(from url: URL) throws -> (Int, MLXArray) {
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw NSError(
            domain: "Qwen3TTS.AudioUtils", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    try audioFile.read(into: buffer)

    guard let floatChannelData = buffer.floatChannelData else {
        throw NSError(
            domain: "Qwen3TTS.AudioUtils", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
    }

    let sampleRate = Int(format.sampleRate)
    let samples = Array(
        UnsafeBufferPointer(start: floatChannelData[0], count: Int(buffer.frameLength)))
    let audioData = MLXArray(samples)

    return (sampleRate, audioData)
}

/// Save audio data to a WAV file.
public func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let samples = audio.asArray(Float.self)

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)

    let frameCount = AVAudioFrameCount(samples.count)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw NSError(
            domain: "Qwen3TTS.AudioUtils", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    buffer.frameLength = frameCount

    if let channelData = buffer.floatChannelData {
        for i in 0..<samples.count {
            channelData[0][i] = samples[i]
        }
    }

    try audioFile.write(from: buffer)
}
