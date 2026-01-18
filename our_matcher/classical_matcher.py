import torch
import torch.nn.functional as F
import time


class ClassicalMatcher:
    """
    Custom feature matcher using brute-force matching with Lowe's ratio test.
    This is a classical approach vs LightGlue's neural network approach.
    """

    def __init__(self, ratio_threshold=0.75, mutual_check=True):
        """
        Args:
            ratio_threshold: Lowe's ratio test threshold (default 0.75)
            mutual_check: Use mutual nearest neighbor check (default True)
        """
        self.ratio_threshold = ratio_threshold
        self.mutual_check = mutual_check

    def match(self, feats0, feats1):
        """
        Match features between two images using brute-force + Lowe's ratio test.

        Args:
            feats0: Features from image 0 (dict with 'descriptors' and 'keypoints')
            feats1: Features from image 1 (dict with 'descriptors' and 'keypoints')

        Returns:
            matches: Tensor of shape (N, 2) with matched indices
            scores: Confidence scores for each match
        """
        start_time = time.time()

        # Extract descriptors (shape: [1, N, 256])
        desc0 = feats0['descriptors'][0]  # [N0, 256]
        desc1 = feats1['descriptors'][0]  # [N1, 256]

        # Compute pairwise distances (cosine similarity)
        # Normalize descriptors
        desc0_norm = F.normalize(desc0, p=2, dim=1)
        desc1_norm = F.normalize(desc1, p=2, dim=1)

        # Compute similarity matrix: [N0, N1]
        similarity = torch.matmul(desc0_norm, desc1_norm.t())

        # Convert to distance (1 - similarity for cosine distance)
        distance = 1.0 - similarity

        # Find two nearest neighbors for each descriptor in desc0
        # distances0: [N0, 2], indices0: [N0, 2]
        distances0, indices0 = torch.topk(distance, k=2, dim=1, largest=False)

        # Apply Lowe's ratio test
        ratio = distances0[:, 0] / (distances0[:, 1] + 1e-8)
        valid_mask = ratio < self.ratio_threshold

        # Get tentative matches
        matches0to1 = indices0[:, 0]  # Best match for each desc0

        if self.mutual_check:
            # Perform mutual nearest neighbor check
            distances1, indices1 = torch.topk(distance.t(), k=1, dim=1, largest=False)
            matches1to0 = indices1[:, 0]

            # Check mutual consistency
            mutual_mask = matches1to0[matches0to1] == torch.arange(len(desc0), device=desc0.device)
            valid_mask = valid_mask & mutual_mask

        # Extract valid matches
        idx0 = torch.where(valid_mask)[0]
        idx1 = matches0to1[valid_mask]

        matches = torch.stack([idx0, idx1], dim=1)

        # Compute confidence scores (inverse of distance, normalized)
        match_distances = distances0[valid_mask, 0]
        scores = 1.0 - match_distances

        elapsed_time = time.time() - start_time

        return {
            'matches': matches,
            'scores': scores,
            'matching_time': elapsed_time
        }

    def __call__(self, data):
        """
        Interface compatible with LightGlue for easy comparison.
        """
        return self.match(data['image0'], data['image1'])
