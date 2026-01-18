# LightGlue vs Classical Feature Matching: A Comparative Study

## Project Overview

This project presents a comprehensive comparison between LightGlue, a state-of-the-art transformer-based feature matcher (ICCV 2023), and a classical implementation using brute-force matching with Lowe's ratio test. The objective is to analyze trade-offs between matching completeness, geometric accuracy, and computational efficiency on identical feature sets.

## Abstract

Feature matching is fundamental to numerous computer vision tasks including Structure from Motion (SfM), Visual SLAM, and panorama stitching. While recent neural approaches like LightGlue demonstrate superior completeness through learned attention mechanisms, classical methods remain computationally efficient. This study quantifies these trade-offs using SuperPoint features on real-world image pairs, achieving a 54.9x speedup with competitive accuracy (65.9% vs 66.5% inlier ratio).

## Project Structure

LightGlue/
├── lightglue/ # Official LightGlue implementation
│ ├── init.py
│ ├── lightglue.py
│ ├── superpoint.py
│ └── utils.py
├── our_matcher/ # Custom classical matcher implementation
│ ├── init.py
│ ├── classical_matcher.py # Brute-force + Lowe's ratio test
│ └── evaluator.py # Geometric verification and metrics
├── experiments/ # Comparison and analysis scripts
│ ├── compare_methods.py # Main comparison pipeline
│ ├── visualize_comparison.py # Side-by-side visualization
│ └── poster_charts.py # Quantitative charts generation
├── tests/ # Unit tests and debugging scripts
│ ├── test_gpu.py
│ ├── run_matching.py
│ └── visualize_matches.py
├── assets/ # Test images
│ ├── DSC_0410.JPG
│ └── DSC_0411.JPG
├── results/ # Output visualizations and data
│ ├── comparison_results.json
│ ├── poster_comparison_final.png
│ └── poster_charts_metrics.png
└── README.md

## Methodology

### Pipeline Architecture

1. **Feature Extraction (SuperPoint)**
   - Extract 2048 keypoints per image
   - 256-dimensional descriptors
   - Identical features for both matchers (fair comparison)

2. **Matching Stage**

   **LightGlue (Neural Approach)**
   - Transformer-based architecture
   - Self-attention for feature refinement
   - Cross-attention for correspondence
   - Adaptive depth/width pruning

   **Classical Matcher (Our Implementation)**
   - Cosine similarity distance matrix computation
   - Lowe's ratio test (threshold: 0.75)
   - Mutual nearest neighbor verification
   - Brute-force comparison of all descriptor pairs

3. **Evaluation**
   - RANSAC homography estimation
   - Inlier counting (3-pixel reprojection threshold)
   - Processing time measurement with GPU synchronization
   - Confidence score analysis

## Quantitative Results

### Performance Comparison

| Metric                  | LightGlue | Our Classical Matcher | Difference       |
|-------------------------|-----------|----------------------|------------------|
| Total Matches           | 1,063     | 787                  | -26.0%           |
| Geometric Inliers       | 707       | 519                  | -26.6%           |
| Inlier Ratio            | 66.5%     | 65.9%                | -0.6 pp          |
| Processing Time (ms)    | 278.0     | 5.1                  | **54.9x faster** |
| Throughput (FPS)        | 3.6       | 196.1                | **54.5x higher** |
| Mean Confidence         | 0.901     | 0.908                | +0.8%            |
| Median Confidence       | 0.989     | 0.919                | -7.1%            |

**Hardware:** NVIDIA RTX 3060 Laptop GPU (6GB VRAM), PyTorch 2.7.1 + CUDA 11.8

### Key Findings

1. **Completeness**: LightGlue identifies 35% more correspondences through global context understanding
2. **Accuracy**: Both methods achieve statistically equivalent geometric accuracy (66.5% vs 65.9%)
3. **Speed**: Classical approach achieves 54.9x speedup, enabling real-time processing at 196 FPS
4. **Trade-offs**: Choice depends on application requirements (real-time vs maximum completeness)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- PyTorch 2.0+ with CUDA support

### Setup

```bash
# Clone the repository
git clone https://github.com/salah-eddine-lagrou/LightGlue.git
cd LightGlue

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib numpy seaborn

# Install LightGlue
pip install -e .
