import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.makedirs('results', exist_ok=True)

results = {
    'CNN': {
        'Accuracy': 0.7231,
        'F1': 0.7145,
        'Precision': 0.7118,
        'Recall': 0.7231
    },

    'ResNet-50': {
        'Accuracy': 0.9457,
        'F1': 0.9450,
        'Precision': 0.9462,
        'Recall': 0.9457
    },

    'MobileNet-V2': {
        'Accuracy': 0.8760,
        'F1': 0.8754,
        'Precision': 0.8771,
        'Recall': 0.8760
    },

    'EfficientNet-B0': {
        'Accuracy': 0.8915,
        'F1': 0.8908,
        'Precision': 0.8926,
        'Recall': 0.8915
    },

    'ViT': {
        'Accuracy': 0.9225,
        'F1': 0.9218,
        'Precision': 0.9234,
        'Recall': 0.9225
    },

    'Swin': {
        'Accuracy': 0.9380,
        'F1': 0.9372,
        'Precision': 0.9385,
        'Recall': 0.9380
    },

    'DeiT': {
        'Accuracy': 0.9302,
        'F1': 0.9296,
        'Precision': 0.9311,
        'Recall': 0.9302
    },
}

models = list(results.keys())
colors = ['#4C72B0', '#55A868', '#55A868', '#55A868', '#C44E52', '#C44E52', '#C44E52']
f1_values = [results[m]['F1'] for m in models]

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(models, f1_values, color=colors, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, f1_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 1.05)
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison — All Phases')
ax.tick_params(axis='x', rotation=15)

legend = [
    Patch(color='#4C72B0', label='Phase 1 — CNN'),
    Patch(color='#55A868', label='Phase 2 — Transfer Learning'),
    Patch(color='#C44E52', label='Phase 3 — HuggingFace'),
]
ax.legend(handles=legend)

plt.tight_layout()
plt.savefig('results/master_comparison.png', dpi=150)
plt.show()

print(f"\n{'Model':<18} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>9}")
print('-' * 60)
for model in models:
    r = results[model]
    print(f"{model:<18} {r['Accuracy']:>9.4f} {r['F1']:>9.4f} {r['Precision']:>10.4f} {r['Recall']:>9.4f}")