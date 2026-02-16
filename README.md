# swift-qwen3-tts

Qwen3 TTS for Apple Silicon - a standalone Swift package for text-to-speech synthesis using [MLX](https://github.com/ml-explore/mlx-swift).

Ported from the Python [mlx-audio](https://github.com/Blaizzy/mlx-audio) implementation.

**Platforms**: macOS 14+ / iOS 17+ (Apple Silicon)

> **Prerequisites**: Before running, you need to copy `default.metallib` from your macOS system to the project root directory.
> ```bash
> # Find and copy default.metallib
> cp $(find /usr/lib /System/Library -name "default.metallib" 2>/dev/null | head -1) .
> # Or from the MLX Swift build output:
> cp .build/release/default.metallib .
> ```

## Features

- **VoiceDesign** - Create any voice from a text description (e.g. "A warm female voice")
- **CustomVoice** - Use built-in speakers with emotion/style control
- **Base** - Simple generation with built-in speakers
- **Voice Cloning** - Clone a voice from a 3-second audio reference (Base model)
- **Streaming** - Async stream API for real-time audio generation
- **12 Languages** - Chinese, English, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian + Beijing/Sichuan dialects

## Supported Models

| Model | Size | Type | Speakers |
|-------|------|------|----------|
| [Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16) | 1.7B | VoiceDesign | Any (text-described) |
| [Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16) | 0.6B | CustomVoice | Aiden, Ryan, Serena, Vivian, Sohee, Ono_anna, Uncle_fu, Eric, Dylan |
| [Qwen3-TTS-12Hz-1.7B-Base-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16) | 1.7B | Base + Cloning | Built-in + voice cloning |

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/AtomGradient/swift-qwen3-tts.git", branch: "main"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "Qwen3TTS", package: "swift-qwen3-tts"),
        ]
    ),
]
```

## Quick Start

### VoiceDesign - Describe any voice

```swift
import Qwen3TTS

let model = try await Qwen3TTSModel.fromPretrained(
    "/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
)

let audio = try await model.generate(
    text: "Hello, welcome to Qwen3 TTS!",
    instruct: "A warm female voice with a friendly tone",
    language: "english"
)

// audio is an MLXArray of Float samples at 24kHz
try saveAudioArray(audio, sampleRate: 24000, to: URL(fileURLWithPath: "output.wav"))
```

### CustomVoice - Built-in speaker + emotion

```swift
let model = try await Qwen3TTSModel.fromPretrained(
    "/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
)

let audio = try await model.generate(
    text: "This is exciting news!",
    speaker: "Aiden",
    instruct: "Happy and energetic",
    language: "english"
)
```

### Voice Cloning (Base model)

```swift
let model = try await Qwen3TTSModel.fromPretrained(
    "/path/to/Qwen3-TTS-12Hz-1.7B-Base-bf16"
)

// Load 3+ seconds of reference audio (24kHz)
let (sampleRate, refAudio) = try loadAudioArray(from: URL(fileURLWithPath: "reference.wav"))

let audio = try model.generateVoiceClone(
    text: "New content in the cloned voice.",
    referenceAudio: refAudio,
    referenceText: "Transcript of the reference audio.",
    language: "english"
)
```

### Streaming

```swift
for try await event in model.generateStream(
    text: "Streaming generation example.",
    instruct: "A calm narrator voice"
) {
    switch event {
    case .token(let id):
        break  // codec token generated
    case .info(let info):
        print(info.summary)  // generation stats
    case .audio(let audio):
        // final audio MLXArray
        break
    }
}
```

## CLI Tool

The package includes a command-line demo. Download models from [HuggingFace](https://huggingface.co/mlx-community) and provide the local path:

```bash
# Build
swift build -c release

# VoiceDesign mode
swift run -c release Qwen3TTSDemo \
  --model /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \
  --text "Hello world" \
  --instruct "A clear female voice" \
  --output output.wav

# CustomVoice mode
swift run -c release Qwen3TTSDemo \
  --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16 \
  --text "Hello world" \
  --speaker Ryan \
  --instruct "Calm and professional" \
  --output output.wav

# Voice Cloning (Base model)
swift run -c release Qwen3TTSDemo \
  --model /path/to/Qwen3-TTS-12Hz-1.7B-Base-bf16 \
  --text "New content" \
  --reference-audio reference.wav \
  --reference-text "Reference transcript" \
  --output output.wav
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text`, `-t` | Text to synthesize | `"Hello world"` |
| `--instruct`, `-i` | Voice style description | - |
| `--speaker`, `-s` | Speaker name (CustomVoice/Base) | - |
| `--model`, `-m` | Local path to model directory | (required) |
| `--output`, `-o` | Output WAV file path | `output.wav` |
| `--language`, `-l` | Language code | `auto` |
| `--temperature` | Sampling temperature | `0.9` |
| `--top-k` | Top-K sampling | `50` |
| `--max-tokens` | Max generation tokens | `2048` |
| `--reference-audio` | Reference audio for cloning | - |
| `--reference-text` | Reference audio transcript | - |

## API Reference

### Qwen3TTSModel

```swift
public class Qwen3TTSModel: Module {
    // Load model from a local directory containing config.json and safetensors files
    public static func fromPretrained(_ modelPath: String) async throws -> Qwen3TTSModel

    // Unified generation (auto-routes by model type and parameters)
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
    ) async throws -> MLXArray

    // Streaming generation
    public func generateStream(/* same params */) -> AsyncThrowingStream<Qwen3TTSGeneration, Error>

    // Voice cloning (Base model only)
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
    ) throws -> MLXArray

    // Properties
    public var sampleRate: Int             // 24000
    public var ttsModelType: String        // "voice_design" / "custom_voice" / "base"
    public var supportedSpeakers: [String] // Available speaker names
    public var supportsVoiceCloning: Bool  // Whether model supports cloning
}
```

### Audio Utilities

```swift
// Load audio file to MLXArray
public func loadAudioArray(from url: URL) throws -> (Int, MLXArray)  // (sampleRate, samples)

// Save MLXArray to WAV file
public func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws
```

## Architecture

```
Qwen3 TTS Pipeline
====================

Text Input
    |
    v
[Tokenizer] --> token IDs
    |
    v
[Talker] (28-layer Transformer, MRoPE)
    |  Generates 1st codebook (semantic)
    v
[CodePredictor] (5-layer Transformer)
    |  Generates remaining 15 codebooks (acoustic)
    v
16 Codebook Tokens  [1, seq_len, 16]
    |
    v
[SpeechTokenizer Decoder]
    |  Split RVQ --> Transformer --> Upsample 1920x --> Audio
    v
Audio Output  [samples] @ 24kHz
```

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (0.29.0+) - Apple Silicon ML framework
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (2.29.0+) - LM utilities
- [swift-transformers](https://github.com/huggingface/swift-transformers) (1.0.0+) - Tokenizers


## Acknowledgements

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Original Python implementation by Prince Canuma
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Model by Alibaba Qwen Team
- [MLX](https://github.com/ml-explore/mlx) - Apple Machine Learning framework

## License

MIT
