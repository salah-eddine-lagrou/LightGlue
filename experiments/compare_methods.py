from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import sys

sys.path.append('..')
from our_matcher.classical_matcher import ClassicalMatcher
from our_matcher.evaluator import MatchingEvaluator
import torch
import time
import json


def compare_on_image_pair(image_path0, image_path1, device='cuda'):
    """
    Compare LightGlue vs Custom Matcher on a single image pair.
    """
    print(f"\n{'=' * 70}")
    print(f"Comparing methods on: {image_path0.split('/')[-1]} vs {image_path1.split('/')[-1]}")
    print(f"{'=' * 70}\n")

    # Load images
    image0 = load_image(image_path0).to(device)
    image1 = load_image(image_path1).to(device)

    # Initialize feature extractor (shared by both methods)
    print("Initializing SuperPoint extractor...")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # Extract features once (same for both methods)
    print("Extracting features...")
    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

    print(f"  Image 0: {feats0['keypoints'].shape[1]} keypoints")
    print(f"  Image 1: {feats1['keypoints'].shape[1]} keypoints")

    results = {}

    # ========== Method 1: LightGlue ==========
    print("\n[1/2] Running LightGlue...")
    matcher_lg = LightGlue(features='superpoint').eval().to(device)

    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        matches_lg = matcher_lg({'image0': feats0, 'image1': feats1})

    if device == 'cuda':
        torch.cuda.synchronize()
    time_lg = time.time() - start

    matches_lg = rbd(matches_lg)

    # Evaluate LightGlue
    feats0_rbd, feats1_rbd = rbd(feats0), rbd(feats1)
    metrics_lg = MatchingEvaluator.evaluate_matches(
        feats0_rbd, feats1_rbd,
        matches_lg['matches'],
        matches_lg['scores']
    )
    metrics_lg['time'] = time_lg

    print(f"  ✓ Matches: {metrics_lg['num_matches']}")
    print(f"  ✓ Inliers: {metrics_lg['num_inliers']} ({metrics_lg['inlier_ratio']:.1%})")
    print(f"  ✓ Time: {time_lg * 1000:.1f} ms")

    results['lightglue'] = metrics_lg

    # ========== Method 2: Your Custom Matcher ==========
    print("\n[2/2] Running Custom Matcher...")
    matcher_custom = ClassicalMatcher(ratio_threshold=0.75, mutual_check=True)

    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        matches_custom = matcher_custom({'image0': feats0, 'image1': feats1})

    if device == 'cuda':
        torch.cuda.synchronize()
    time_custom = time.time() - start

    # Evaluate Custom Matcher
    metrics_custom = MatchingEvaluator.evaluate_matches(
        feats0_rbd, feats1_rbd,
        matches_custom['matches'],
        matches_custom['scores']
    )
    metrics_custom['time'] = time_custom

    print(f"  ✓ Matches: {metrics_custom['num_matches']}")
    print(f"  ✓ Inliers: {metrics_custom['num_inliers']} ({metrics_custom['inlier_ratio']:.1%})")
    print(f"  ✓ Time: {time_custom * 1000:.1f} ms")

    results['custom'] = metrics_custom

    # ========== Comparison Summary ==========
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Metric':<25} {'LightGlue':<20} {'Custom':<20} {'Winner'}")
    print(f"{'-' * 70}")

    # Number of matches
    winner = 'LightGlue' if metrics_lg['num_matches'] > metrics_custom['num_matches'] else 'Custom'
    print(f"{'Total Matches':<25} {metrics_lg['num_matches']:<20} {metrics_custom['num_matches']:<20} {winner}")

    # Inliers
    winner = 'LightGlue' if metrics_lg['num_inliers'] > metrics_custom['num_inliers'] else 'Custom'
    print(f"{'Geometric Inliers':<25} {metrics_lg['num_inliers']:<20} {metrics_custom['num_inliers']:<20} {winner}")

    # Inlier ratio
    winner = 'LightGlue' if metrics_lg['inlier_ratio'] > metrics_custom['inlier_ratio'] else 'Custom'
    lg_inlier_str = f"{metrics_lg['inlier_ratio']:.1%}"
    custom_inlier_str = f"{metrics_custom['inlier_ratio']:.1%}"
    print(f"{'Inlier Ratio':<25} {lg_inlier_str:<20} {custom_inlier_str:<20} {winner}")

    # Speed
    winner = 'LightGlue' if time_lg < time_custom else 'Custom'
    speedup = time_custom / time_lg if time_lg < time_custom else time_lg / time_custom
    lg_time_str = f"{time_lg * 1000:.1f}"
    custom_time_str = f"{time_custom * 1000:.1f}"
    print(f"{'Speed (ms)':<25} {lg_time_str:<20} {custom_time_str:<20} {winner} ({speedup:.1f}x)")

    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    results = compare_on_image_pair(
        '../assets/DSC_0410.JPG',
        '../assets/DSC_0411.JPG',
        device=device
    )

    with open('../results/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to results/comparison_results.json")
