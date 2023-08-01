import os
import csv


def read_last_row_mean_k(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        last_row = None
        for row in reader:
            last_row = row
        if last_row is not None and len(last_row) > 0:
            return last_row[10]
        else:
            return None


def update_eeg_values(eeg_values_file, eeg_file_name, mean_k):
    with open(eeg_values_file, "r") as file:
        rows = list(csv.reader(file))
    for row in rows:
        curr_name = row[0] + ".csv"
        if curr_name == eeg_file_name:
            if len(row) < 3:
                row.append("")
            row[2] = mean_k
            break

    with open(eeg_values_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def update_eeg_value(mean_k):
    with open(eeg_values_file, "r") as file:
        rows = list(csv.reader(file))
    for row in rows:
        if len(row) < 3:
            row.append("")
        row[2] = mean_k
        break

    with open(eeg_values_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def process_data_files(data_dir, eeg_values_file):
    with open(eeg_values_file, "r") as file:
        reader = csv.reader(file)
        header = True
        for row in reader:
            if header:
                header = False
                update_eeg_value("mean_k")
                continue
            eeg_file_name = row[0] + ".csv"
            csv_file_name = os.path.join(data_dir, eeg_file_name)
            if os.path.isfile(csv_file_name):
                mean_k = read_last_row_mean_k(csv_file_name)
                if mean_k is not None:
                    update_eeg_values(eeg_values_file, eeg_file_name, mean_k)

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_files_dir = os.path.join(parent_dir, "data_files")
eeg_values_file = os.path.join(parent_dir, "eeg_values.csv")
process_data_files(data_files_dir, eeg_values_file)
