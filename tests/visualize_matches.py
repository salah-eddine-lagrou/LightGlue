from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading models...")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

print("Loading images...")
image0 = load_image('../assets/sacre_coeur1.jpg').to(device)
image1 = load_image('../assets/sacre_coeur2.jpg').to(device)

print("Extracting and matching...")
with torch.no_grad():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})

# Remove batch dimension
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

# Get matched points
matches = matches01['matches'].cpu().numpy()
points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
scores = matches01['scores'].cpu().numpy()

# Convert images to numpy for visualization
img0 = image0.cpu().permute(1, 2, 0).numpy()
img1 = image1.cpu().permute(1, 2, 0).numpy()

# Create side-by-side image
h0, w0 = img0.shape[:2]
h1, w1 = img1.shape[:2]
h_max = max(h0, h1)
combined = np.ones((h_max, w0 + w1, 3))
combined[:h0, :w0] = img0
combined[:h1, w0:w0 + w1] = img1

print(f"\nCreating visualization with {len(matches)} matches...")

# Plot - showing top 100 matches by confidence
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.imshow(combined)

# Sort by confidence and take top 100
top_indices = np.argsort(scores)[-100:]

for idx in top_indices:
    pt0 = points0[idx]
    pt1 = points1[idx]
    score = scores[idx]

    # Color based on confidence (green = high, yellow = medium, red = low)
    color = plt.cm.RdYlGn(score)

    # Draw line connecting matched points
    ax.plot([pt0[0], pt1[0] + w0], [pt0[1], pt1[1]],
            color=color, linewidth=1, alpha=0.7)

    # Draw keypoints
    ax.plot(pt0[0], pt0[1], 'o', color=color, markersize=5)
    ax.plot(pt1[0] + w0, pt1[1], 'o', color=color, markersize=5)

ax.axis('off')
ax.set_title(f'LightGlue Feature Matching: {len(matches)} total matches (showing top 100 by confidence)',
             fontsize=18, pad=20, weight='bold')

plt.tight_layout()
output_file = 'lightglue_matches_visualization.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✓ Visualization saved to: {output_file}")
print("  Opening image...")
plt.show()
