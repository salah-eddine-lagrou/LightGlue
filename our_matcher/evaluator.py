import torch
import numpy as np
import cv2


class MatchingEvaluator:
    """
    Evaluate and compare feature matching methods.
    """

    @staticmethod
    def compute_homography_inliers(points0, points1, threshold=3.0):
        """
        Compute homography and count inliers (geometrically consistent matches).

        Args:
            points0: Keypoints from image 0 [N, 2]
            points1: Keypoints from image 1 [N, 2]
            threshold: RANSAC reprojection threshold in pixels

        Returns:
            num_inliers: Number of geometrically consistent matches
            inlier_ratio: Ratio of inliers to total matches
        """
        if len(points0) < 4:
            return 0, 0.0

        # Convert to numpy if needed
        if torch.is_tensor(points0):
            points0 = points0.cpu().numpy()
        if torch.is_tensor(points1):
            points1 = points1.cpu().numpy()

        # Find homography using RANSAC
        H, mask = cv2.findHomography(points0, points1, cv2.RANSAC, threshold)

        if mask is None:
            return 0, 0.0

        num_inliers = int(mask.sum())
        inlier_ratio = num_inliers / len(points0)

        return num_inliers, inlier_ratio

    @staticmethod
    def evaluate_matches(feats0, feats1, matches, scores):
        """
        Evaluate match quality.

        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Extract matched points
        kpts0 = feats0['keypoints'][0] if feats0['keypoints'].dim() == 3 else feats0['keypoints']
        kpts1 = feats1['keypoints'][0] if feats1['keypoints'].dim() == 3 else feats1['keypoints']

        points0 = kpts0[matches[:, 0]]
        points1 = kpts1[matches[:, 1]]

        # Geometric verification
        num_inliers, inlier_ratio = MatchingEvaluator.compute_homography_inliers(
            points0, points1
        )

        metrics = {
            'num_matches': len(matches),
            'num_inliers': num_inliers,
            'inlier_ratio': inlier_ratio,
            'mean_confidence': scores.mean().item(),
            'median_confidence': scores.median().item(),
            'min_confidence': scores.min().item(),
            'max_confidence': scores.max().item(),
        }

        return metrics
