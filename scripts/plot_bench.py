import matplotlib.pyplot as plt
import numpy as np

# Data
cases = ['IEEE 39', 'IEEE 118', 'PEGASE 9241']
pandapower_times = [38.9, 42.8, 145.5]
lightsim_times = [0.12, 0.35, 51.2]
rustpower_times = [0.04, 0.10, 30.5]

x = np.arange(len(cases))
width = 0.25

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Full Scale Plot
rects1 = ax1.bar(x - width, pandapower_times, width, label='pandapower 3', color='#aec7e8')
rects2 = ax1.bar(x, lightsim_times, width, label='LightSim2Grid (KLU)', color='#ffbb78')
rects3 = ax1.bar(x + width, rustpower_times, width, label='rustpower (KLU)', color='#98df8a')

ax1.set_ylabel('Time (ms)')
ax1.set_title('Performance Comparison (Lower is Better - Linear Scale)')
ax1.set_xticks(x)
ax1.set_xticklabels(cases)
ax1.legend()
ax1.grid(True, axis='y', ls="-", alpha=0.3)

# Improve Speedup Annotations: Place them clearly above the Pandapower baseline
y_max = max(pandapower_times) * 1.3
ax1.set_ylim(0, y_max)

for i in range(len(cases)):
    speedup = pandapower_times[i] / rustpower_times[i]
    # Place above the pandapower bar for clear context
    ax1.annotate(f'{speedup:.1f}x faster',
                 xy=(x[i] - width, pandapower_times[i]),
                 xytext=(30, 20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#1f77b4'),
                 ha='center', va='bottom', color='#1f77b4', fontweight='bold', fontsize=9)

# Focus on PEGASE 9241 (Linear Scale)
case_idx = 2
bars2 = ax2.bar(['pandapower 3', 'LightSim2Grid (KLU)', 'rustpower (KLU)'], 
        [pandapower_times[case_idx], lightsim_times[case_idx], rustpower_times[case_idx]],
        color=['#aec7e8', '#ffbb78', '#98df8a'])
ax2.set_ylabel('Time (ms)')
ax2.set_title(f'Focus: {cases[case_idx]} Core Solve Time')
ax2.grid(axis='y', ls="-", alpha=0.5)

# Adding value labels with collision avoidance
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        # Only show labels if they won't cluster at the bottom
        label_text = f'{height:.2f}' if height > 0.5 else f'{height:.3f}'
        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax1)
autolabel(bars2, ax2)

plt.tight_layout()
plt.savefig('docs/performance_comparison.png', dpi=300)
print("Benchmark plot saved to docs/performance_comparison.png")
