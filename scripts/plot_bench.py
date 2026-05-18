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
# ax1.set_yscale('log')  # Removed log scale - it was too kind!
ax1.grid(True, axis='y', ls="-", alpha=0.3)

# Adding speedup annotations
for i in range(len(cases)):
    speedup = pandapower_times[i] / rustpower_times[i]
    ax1.text(x[i] + width, rustpower_times[i] + 2, f'{speedup:.1f}x', 
             ha='center', va='bottom', color='green', fontweight='bold', fontsize=9)

# Focus on PEGASE 9241 (Linear Scale)
case_idx = 2
ax2.bar(['pandapower 3', 'LightSim2Grid (KLU)', 'rustpower (KLU)'], 
        [pandapower_times[case_idx], lightsim_times[case_idx], rustpower_times[case_idx]],
        color=['#aec7e8', '#ffbb78', '#98df8a'])
ax2.set_ylabel('Time (ms)')
ax2.set_title(f'Focus: {cases[case_idx]} Core Solve Time')
ax2.grid(axis='y', ls="-", alpha=0.5)

# Adding value labels
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax1)

plt.tight_layout()
plt.savefig('docs/performance_comparison.png', dpi=300)
print("Benchmark plot saved to docs/performance_comparison.png")
