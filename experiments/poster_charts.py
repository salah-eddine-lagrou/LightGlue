import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Load your results
with open('../results/comparison_results.json', 'r') as f:
    results = json.load(f)

lg = results['lightglue']
custom = results['custom']

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
fig = plt.figure(figsize=(20, 16))

# Chart 1: Number of Matches Bar Chart
ax1 = plt.subplot(2, 3, 1)
methods = ['LightGlue', 'Our Method']
matches = [lg['num_matches'], custom['num_matches']]
bars1 = plt.bar(methods, matches, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Total Matches Found', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of Matches', fontsize=14)
plt.ylim(0, max(matches)*1.1)
for bar, val in zip(bars1, matches):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Chart 2: Geometric Accuracy (Inlier Ratio)
ax2 = plt.subplot(2, 3, 2)
inlier_ratios = [lg['inlier_ratio']*100, custom['inlier_ratio']*100]
bars2 = plt.bar(methods, inlier_ratios, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Geometric Accuracy\n(Inlier Ratio %)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Inlier Ratio (%)', fontsize=14)
plt.ylim(0, 100)
for bar, val in zip(bars2, inlier_ratios):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Chart 3: Processing Speed
ax3 = plt.subplot(2, 3, 3)
times_ms = [lg['time']*1000, custom['time']*1000]
bars3 = plt.bar(methods, times_ms, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Processing Speed', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Time (ms)', fontsize=14)
plt.ylim(0, max(times_ms)*1.1)
for bar, val in zip(bars3, times_ms):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.1f}ms', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Chart 4: Speedup Factor
ax4 = plt.subplot(2, 3, 4)
speedup = lg['time'] / custom['time']
plt.bar(['Speedup Factor'], [speedup], color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=3)
plt.title(f'Speed Improvement\n{lg["time"]/custom["time"]:.1f}x Faster!', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Speedup Multiple', fontsize=14)
plt.ylim(0, speedup*1.1)
plt.text(0, speedup + 0.5, f'{speedup:.1f}x', ha='center', va='bottom', fontsize=20, fontweight='bold', color='darkred')

# Chart 5: Confidence Distribution
ax5 = plt.subplot(2, 3, 5)
confidences = {
    'LightGlue': [lg['mean_confidence'], lg['median_confidence']],
    'Our Method': [custom['mean_confidence'], custom['median_confidence']]
}
x = np.arange(len(confidences['LightGlue']))
width = 0.35
p1 = plt.bar(x - width/2, confidences['LightGlue'], width, label='LightGlue',
             color='#2E86AB', alpha=0.8, edgecolor='black')
p2 = plt.bar(x + width/2, confidences['Our Method'], width, label='Our Method',
             color='#A23B72', alpha=0.8, edgecolor='black')
plt.title('Match Confidence', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Confidence Score', fontsize=14)
plt.xticks(x, ['Mean', 'Median'])
plt.legend()
plt.ylim(0, 1)

# Chart 6: Inliers Absolute Number
ax6 = plt.subplot(2, 3, 6)
inliers = [lg['num_inliers'], custom['num_inliers']]
bars6 = plt.bar(methods, inliers, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Absolute Number of Inliers', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of Inliers', fontsize=14)
plt.ylim(0, max(inliers)*1.1)
for bar, val in zip(bars6, inliers):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.suptitle('LightGlue vs Our Classical Matcher: Quantitative Comparison\n'
             '(RTX 3060 Laptop GPU | SuperPoint Features)', fontsize=24, fontweight='bold', y=0.98)

plt.tight_layout()

# Save high-quality poster charts
plt.savefig('../results/poster_charts_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('../results/poster_charts_metrics.jpg', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Files:")
print("  ../results/poster_charts_metrics.png")
print("  ../results/poster_charts_metrics.jpg")
