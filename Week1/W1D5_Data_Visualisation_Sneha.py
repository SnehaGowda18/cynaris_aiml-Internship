# W1D5: Data Visualisation - Matplotlib & Seaborn

import os
import matplotlib.pyplot as plt
import seaborn as sns


def practice(values):

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # Create index values
    x = range(len(values))


    # Matplotlib Line Plot
    plt.figure(figsize=(6,4))

    plt.plot(x, values, marker='o')

    plt.title("Value Trend using Matplotlib")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.grid(True)

    # Save image
    plt.savefig("outputs/matplotlib_plot.png")
    plt.close()



    # Seaborn Distribution Plot
    plt.figure(figsize=(6,4))

    sns.histplot(values, kde=True)

    plt.title("Data Distribution using Seaborn")
    plt.xlabel("Values")
    plt.ylabel("Frequency")

    # Save image
    plt.savefig("outputs/seaborn_distribution.png")
    plt.close()



# Test Case 1
practice([10, 20, 30, 40, 50])

# Test Case 2
practice([5, 15, 25, 35, 45])

# Test Case 3
practice([100, 80, 60, 40, 20])


print("Done! Review with CIA for feedback.")