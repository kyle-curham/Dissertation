import numpy as np
import matplotlib.pyplot as plt

means2 = []
maxs = []
medians2 = []
for i in range(1000):
    # Generate 7 samples from a uniform distribution between 1 and 78
    samples = np.random.uniform(low=1, high=78, size=7)
    mean2 = 2*np.mean(samples)
    means2.append(mean2)
    max = np.max(samples)
    maxs.append(max)

# Print the results of the first 10 samples
print("double mean: ", means2[:10])
print("max: ", maxs)

# Plot the means
plt.hist(means2, bins=10)
plt.show()

plt.hist(maxs, bins=10)
plt.show()


# Simple binomial sampling
binomial_samples = np.random.binomial(n=20, p=0.5, size=1000)
print("\nBinomial samples (n=20, p=0.5):", binomial_samples)

plt.hist(binomial_samples, bins=10)
plt.show()
