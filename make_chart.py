import matplotlib.pyplot as plt
import numpy as np

labels = ['AAPL 2024\n(bull)', 'AAPL 2022\n(bear)', 'JPM 2024\n(bull)', 'JPM 2022\n(bear)']
agent_returns = [-6.45, -4.13, 17.12, -7.46]
bh_returns = [-7.51, -2.21, 17.12, -12.56]
agent_dd = [-6.45, -8.44, -2.30, -9.59]
bh_dd = [-13.00, -12.21, -2.30, -20.25]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(labels))
w = 0.35

ax1.bar(x - w/2, agent_returns, w, label='Agents', color='#2E86AB')
ax1.bar(x + w/2, bh_returns, w, label='Buy & Hold', color='#A8B5C7')
ax1.set_title('Cumulative Return by Regime', fontsize=13, fontweight='bold')
ax1.set_ylabel('Return (%)')
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.legend(); ax1.grid(axis='y', alpha=0.3)

ax2.bar(x - w/2, agent_dd, w, label='Agents', color='#2E86AB')
ax2.bar(x + w/2, bh_dd, w, label='Buy & Hold', color='#A8B5C7')
ax2.set_title('Maximum Drawdown by Regime', fontsize=13, fontweight='bold')
ax2.set_ylabel('Drawdown (%)')
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

plt.suptitle('TradingAgents: Agent vs Buy & Hold across Regimes',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results_chart.png', dpi=150, bbox_inches='tight')
print("Saved to results_chart.png")