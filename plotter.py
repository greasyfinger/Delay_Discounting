import csv
import matplotlib.pyplot as plt

def plot_eeg_vs_mean_k(csv_file):
    eeg_values = []
    mean_k_values = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            eeg_values.append(float(row['eeg_value']))
            mean_k_values.append(float(row['mean_k']))

    plt.scatter(eeg_values, mean_k_values, marker='o')
    plt.xlabel('EEG Value (10^-9)')
    plt.ylabel('Mean_k')
    plt.title('EEG Value vs. Mean_k')
    plt.grid(True)
    plt.show()

csv_file = "eeg_values.csv"
plot_eeg_vs_mean_k(csv_file)
