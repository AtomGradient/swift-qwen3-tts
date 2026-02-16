// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "swift-qwen3-tts",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        // Core Qwen3 TTS library
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        ),
        // Command line demo tool
        .executable(
            name: "Qwen3TTSDemo",
            targets: ["Qwen3TTSDemo"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", from: "2.29.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        // MARK: - Core Library
        .target(
            name: "Qwen3TTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Qwen3TTS",
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-warn-concurrency"], .when(configuration: .debug))
            ]
        ),

        // MARK: - CLI Demo
        .executableTarget(
            name: "Qwen3TTSDemo",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Qwen3TTSDemo"
        ),

        // MARK: - Tests
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/Qwen3TTSTests"
        ),
    ]
)
