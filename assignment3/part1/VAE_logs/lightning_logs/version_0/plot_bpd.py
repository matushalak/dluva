import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSVs
train = pd.read_csv('train_bpd.csv')
val   = pd.read_csv('val_bpd.csv')
test  = pd.read_csv('test_bpd.csv')

# Extract test loss (assuming single-row CSV)
test_loss = test['Value'].iloc[0]

# Set seaborn style (optional)
sns.set(style="whitegrid")

# Create 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ----- Left: train/val loss curves -----
ax0 = axes[0]
sns.lineplot(data=train, x='Step', y='Value', ax=ax0, label='Train')
sns.lineplot(data=val,   x='Step', y='Value', ax=ax0, label='Val')

ax0.set_title('Train & Val Loss')
ax0.set_xlabel('Training step')
ax0.set_ylabel('Loss (bpd)')
ax0.legend()

# ----- Right: barplot of test loss -----
ax1 = axes[1]
sns.barplot(x=['Test-set evaluation after training'], y=[test_loss], ax=ax1, color='green')

ax1.set_title('Test Loss')
ax1.set_ylabel('Loss (bpd)')
ax1.set_xlabel('')

# Tight layout and show
plt.tight_layout()
plt.show()