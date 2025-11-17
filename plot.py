# A simple script to visualize the training log file.

import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE_TO_PLOT = 'model_logs/v17.1.csv'

try:
    data = pd.read_csv(LOG_FILE_TO_PLOT)
except FileNotFoundError:
    exit()

# `figsize` makes the window a nice size. `sharex=True` links the x-axis of both plots.
# Create fig with 2 plots
fig, (plot1, plot2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

fig.suptitle('Training Performance', fontsize=16)

# Plot 1, mean reward
plot1.set_title('Mean Reward')
plot1.plot(data['iteration'], data['reward_mean'], color='blue')
plot1.set_ylabel('Mean Reward')
plot1.grid(True, linestyle='--', alpha=0.6)

# Plot 2, loss
plot2.set_title('Training Loss')
plot2.plot(data['iteration'], data['loss'], color='red')
plot2.set_ylabel('Loss')
plot2.set_xlabel('Iteration')
plot2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()

