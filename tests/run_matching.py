from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch
import time

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Load models
print("Loading SuperPoint and LightGlue models...")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)
print("Models loaded!\n")

# Load images
print("Loading images...")
image0 = load_image('../assets/DSC_0410.JPG').to(device)
image1 = load_image('../assets/DSC_0411.JPG').to(device)
print(f"Image 1 shape: {image0.shape}")
print(f"Image 2 shape: {image1.shape}\n")

# Extract features
print("Extracting keypoints and descriptors...")
start = time.time()

with torch.no_grad():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

print(f"Image 1: {feats0['keypoints'].shape[1]} keypoints detected")
print(f"Image 2: {feats1['keypoints'].shape[1]} keypoints detected\n")

# Match features
print("Matching features with LightGlue...")
with torch.no_grad():
    matches01 = matcher({'image0': feats0, 'image1': feats1})

end = time.time()

# Get results
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
matches = matches01['matches']
scores = matches01['scores']

# Display results
print("\n" + "=" * 60)
print("MATCHING RESULTS")
print("=" * 60)
print(f"Total matches found: {len(matches)}")
print(f"Processing time: {(end - start) * 1000:.1f} ms ({1 / (end - start):.1f} FPS)")
print(f"Average match confidence: {scores.mean():.3f}")
print(f"Confidence range: [{scores.min():.3f}, {scores.max():.3f}]")
print("=" * 60)

# Show sample matches
points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

print(f"\nTop 5 matches (by confidence):")
top5_idx = scores.argsort(descending=True)[:5]
for i, idx in enumerate(top5_idx):
    p0 = points0[idx]
    p1 = points1[idx]
    print(f"  {i + 1}. ({p0[0]:.1f}, {p0[1]:.1f}) -> ({p1[0]:.1f}, {p1[1]:.1f}), confidence: {scores[idx]:.3f}")

print("\n✓ Matching completed successfully!")
