import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function for generating a histogram
def generate_histogram(data, col, title, x_label, y_label):
    """
    Generate a histogram for a specified column.

    Parameters:
    - data: DataFrame containing the necessary columns.
    - col: Column to create a histogram for.
    - title: Title of the histogram.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.hist(data[col], bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Function for generating a cluster map
def generate_cluster_map(data, cmap="viridis", figsize=(11, 11)):
    """
    Generate a cluster map of the correlation matrix.

    Parameters:
    - data: DataFrame containing the necessary columns.
    - cmap: Colormap for the cluster map.
    - figsize: Size of the figure.

    Returns:
    - None
    """
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(correlation_matrix, cmap=cmap, aspect='auto')
    cbar = plt.colorbar(cax)
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(correlation_matrix.columns, fontsize=12)
    ax.set_title('Cluster Map of Correlation Matrix', fontsize=14)
    plt.show()

# Function for preprocessing data
def preprocess_data(data):
    """
    Preprocess the data by mapping values.

    Parameters:
    - data: DataFrame containing the necessary columns.

    Returns:
    - DataFrame: Processed DataFrame.
    """
    data["Survived"] = data["Survived"].map({1: "Survived", 0: "Died"})
    data["Sex"] = data["Sex"].map({"male": "Male", "female": "Female"})
    
    return data

# Function for generating a line plot
def generate_line_plot(data, x_col, y_col, label, marker='o'):
    """
    Generate a line plot.

    Parameters:
    - data: DataFrame containing the data.
    - x_col: Column to be used on the x-axis.
    - y_col: Column to be plotted on the y-axis.
    - label: Label for the line.
    - marker: Marker style for the line. Default is 'o'.
    """
    plt.plot(data[x_col], data[y_col], label=label, marker=marker)
    
    # Set plot labels and title
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.title("Age vs. Fare for Died Passengers")

    # Add legend
    plt.legend()

    # Customize the layout
    plt.gca().set_facecolor("#f2f2f2")
    plt.gcf().patch.set_facecolor("#f2f2f2")
    plt.gca().spines[["left", "bottom"]].set_color('black')
    plt.gca().spines[["left", "bottom"]].set_linewidth(2)
    plt.gca().spines[["left", "bottom"]].set_linestyle("solid")
    plt.gca().xaxis.label.set_color('black')
    plt.gca().yaxis.label.set_color('black')
    plt.gca().title.set_color('black')

    # Show the plot
    plt.show()
# Main Method
def main():
    # Read data from CSV
    df = pd.read_csv('titanic_dataset.csv')
    
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Generate and display a histogram
    generate_histogram(df, 'Age', 'Distribution of Ages', 'Age', 'Frequency')

    # Generate and display a cluster map
    generate_cluster_map(df[numerical_columns], cmap="viridis", figsize=(11, 11))

    # Preprocess the data
    df = preprocess_data(df)

    # Generate line plots for Age vs. Fare for Died passengers
    generate_line_plot(df[df["Survived"] == "Died"], "Age", "Fare", "Died", marker='x')

# Run the main method
if __name__ == "__main__":
    main()
