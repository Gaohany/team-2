import matplotlib.pyplot as plt
import numpy as np

# Define the number of points for each plot
counts = [100, 1000]
distribution_types = ["uniform", "normal"]

# Set the figure size to accommodate the 4 plots
plt.figure(figsize=(14, 8))

# Loop over the counts and distribution_types to create 4 subplots
for j, count in enumerate(counts):
    for i, distribution_type in enumerate(distribution_types):
        # Create subplot position index (subplot starts at 1 not 0)
        ax = plt.subplot(2, 2, i * 2 + j + 1)

        # Generate x and y data points
        if distribution_type == "uniform":
            x = np.random.uniform(0, 1, count)
            y = np.random.uniform(0, 1, count)
        elif distribution_type == "normal":
            x = np.random.normal(0.4, 0.1, count)
            y = np.random.normal(0.3, 0.1, count)
            # Make sure that the points are within the [0, 1] interval
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

        # Plot the points
        plt.scatter(x, y, marker="x")

        # Set the title for each plot
        if i == 0:
            plt.title(f"N={count}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if i == 0:
            plt.xticks([])

        if j == 1:
            plt.yticks([])

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.savefig("density_sparsity.jpg")
