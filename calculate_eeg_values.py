# %matplotlib inline
# !pip install mne
# !pip install meeg_tools
import os
import os.path as op
import mne
import numpy as np
import pandas as pd
from pathlib import Path
# !pip install ipyfilechooser
from ipyfilechooser import FileChooser
from meeg_tools.preprocessing import *
from meeg_tools.utils.epochs import create_epochs
from meeg_tools.utils.raw import read_raw_measurement, filter_raw
from meeg_tools.utils.log import update_log
from mne.epochs import Epochs
from mne.time_frequency import tfr_morlet

def eeg_value(edf_file):
  raw = mne.io.read_raw_edf(edf_file, preload=True)        # Returns a Raw object containing BrainVision data
  raw.load_data()     # Loading continuous data
  settings['bandpass_filter']['low_freq'] = 1
  settings['bandpass_filter']['high_freq'] = 30
  raw_bandpass = filter_raw(raw)
  epochs = create_epochs(raw=raw_bandpass)
  epochs_faster = prepare_epochs_for_ica(epochs=epochs)
  settings['ica']['n_components'] = 20
  settings['ica']['method'] = 'picard'
  ica = mne.preprocessing.ICA(n_components=32, random_state=42)
  ica.fit(epochs_faster)
  epochs_ica=epochs_faster
  montage = mne.channels.make_standard_montage('standard_1005')
  # Get the channel names from the montage
  montage_channel_names = montage.ch_names
  # Get the channel names from the info attribute
  info_channel_names = epochs_ica.info['ch_names']
  # Create a mapping of old channel names to new channel names
  mapping = {info_channel_names[i]: montage_channel_names[i] for i in range(len(info_channel_names))}
  # Rename the channels in the info attribute
  epochs_ica.rename_channels(mapping)
  # Set the montage to the updated info attribute of your epochs_ica object
  epochs_ica.set_montage(montage)
  # # Now, you can run Autoreject with the epochs_ica object
  reject_log = run_autoreject(epochs_ica, n_jobs=11, subset=True)
  epochs_autoreject = epochs_ica.copy().drop(reject_log.report, reason='AUTOREJECT')
  # findining bad channel
  bads = get_noisy_channels(epochs=epochs_autoreject, with_ransac=True)
  def interpolate_bad_channels(epochs, bads):
      epochs.load_data()
      epochs_interpolated = epochs.copy()
      epochs_interpolated.interpolate_bads(reset_bads=True)
      bads_str = ', interpolated: ' + ', '.join(bads)
      if epochs_interpolated.info['description'] is None:
          epochs_interpolated.info['description'] = bads_str
      else:
          epochs_interpolated.info['description'] += bads_str
      return epochs_interpolated
  epochs_ransac = interpolate_bad_channels(epochs=epochs_autoreject, bads=bads)
  # confusion(ask sir): the above get noisy channel function detects that 81 percent of the channels were noisy but when I passed those bad channels to the interpolate bad channel function, it says no channels are noisy
  epochs_ransac.set_eeg_reference('average')
  baseline_start = 0.0
  baseline_end = 1.0
  # performing baseline correction
  epochs_ransac.apply_baseline(baseline=(baseline_start, baseline_end))
  #  Delta Case Frequency
  freqs = np.logspace(*np.log10([1, 4]), num=20)

  channel_of_interest = ['Fz','Oz','Pz','Cz']
  epochs = epochs_ransac.copy().pick_channels(channel_of_interest)

  power, itc = tfr_morlet(epochs,
                          freqs=freqs,
                          n_cycles=freqs/2,
                          return_itc=True,
                          decim=1,
                          average=True,
                          n_jobs=8)
  # simple normalization of power bins by 1/f

  power_f_corrected = np.zeros_like(power.data)

  # divide power by 1/f
  for e in range(power.data.shape[0]):
      for f in range(power.data.shape[1]):
          power_f_corrected[e][f] = power.data[e][f] / (1/power.freqs[f])

  power.data = power_f_corrected
  power_avg_epoch_electrode = power.data.mean(axis=-1)
  power.to_data_frame(picks=None, index=None, long_format=False, time_format='ms')
  case1=np.mean(power_avg_epoch_electrode)
  freqs = np.logspace(*np.log10([4, 8]), num=20)
  # theta frequency case
  channel_of_interest = ['Fz','Oz','Pz','Cz']
  epochs = epochs_ransac.copy().pick_channels(channel_of_interest)

  power, itc = tfr_morlet(epochs,
                          freqs=freqs,
                          n_cycles=freqs/2,
                          return_itc=True,
                          decim=1,
                          average=True,
                          n_jobs=8)
  # simple normalization of power bins by 1/f

  power_f_corrected = np.zeros_like(power.data)

  # divide power by 1/f
  for e in range(power.data.shape[0]):
      for f in range(power.data.shape[1]):
          power_f_corrected[e][f] = power.data[e][f] / (1/power.freqs[f])

  power.data = power_f_corrected
  power_avg_epoch_electrode = power.data.mean(axis=-1)
  power.to_data_frame(picks=None, index=None, long_format=False, time_format='ms')
  case2=np.mean(power_avg_epoch_electrode)
  freqs = np.logspace(*np.log10([8, 13]), num=20)
  # alpha frequency case
  channel_of_interest = ['Fz','Oz','Pz','Cz']
  epochs = epochs_ransac.copy().pick_channels(channel_of_interest)

  power, itc = tfr_morlet(epochs,
                          freqs=freqs,
                          n_cycles=freqs/2,
                          return_itc=True,
                          decim=1,
                          average=True,
                          n_jobs=8)
  # simple normalization of power bins by 1/f

  power_f_corrected = np.zeros_like(power.data)

  # divide power by 1/f
  for e in range(power.data.shape[0]):
      for f in range(power.data.shape[1]):
          power_f_corrected[e][f] = power.data[e][f] / (1/power.freqs[f])

  power.data = power_f_corrected
  power_avg_epoch_electrode = power.data.mean(axis=-1)
  power.to_data_frame(picks=None, index=None, long_format=False, time_format='ms')
  case3=np.mean(power_avg_epoch_electrode)
  freqs = np.logspace(*np.log10([13, 30]), num=20)
  # Beta frequency case
  channel_of_interest = ['Fz','Oz','Pz','Cz']
  epochs = epochs_ransac.copy().pick_channels(channel_of_interest)

  power, itc = tfr_morlet(epochs,
                          freqs=freqs,
                          n_cycles=freqs/2,
                          return_itc=True,
                          decim=1,
                          average=True,
                          n_jobs=8)
  # simple normalization of power bins by 1/f

  power_f_corrected = np.zeros_like(power.data)

  # divide power by 1/f
  for e in range(power.data.shape[0]):
      for f in range(power.data.shape[1]):
          power_f_corrected[e][f] = power.data[e][f] / (1/power.freqs[f])

  power.data = power_f_corrected
  power_avg_epoch_electrode = power.data.mean(axis=-1)
  power.to_data_frame(picks=None, index=None, long_format=False, time_format='ms')
  case4=np.mean(power_avg_epoch_electrode)
  Final_answer= (case1+case2+case3+case4)/4
  return Final_answer

def process_eeg_files_and_save_to_csv():
    src_dir = "eeg_files"
    csv_file_path = "eeg_values.csv"

    # Check if the source directory exists
    if not os.path.exists(src_dir):
        print(f"Source directory '{src_dir}' does not exist.")
        return

    # Initialize an empty dictionary to store the data
    data = {}

    # Get a list of all files in the source directory
    files = os.listdir(src_dir)

    # Process each file and store the results in the data dictionary
    for file in files:
        file_path = os.path.join(src_dir, file)

        # Extract the identifying "name" column from the file name before encountering "_"
        identifying_name = file.split("_")[0]

        # Call the eeg_value() function to get the EEG value
        print(file_path)
        try:
          eeg_result = eeg_value(file_path)
        except:
          eeg_result = -1

        # Store the result in the data dictionary with the identifying name as the key
        data[identifying_name] = eeg_result

    # Create a pandas DataFrame from the data dictionary
    df = pd.DataFrame(list(data.items()), columns=['name', 'eeg_value'])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Data has been processed and saved to '{csv_file_path}'.")
    return df
df = process_eeg_files_and_save_to_csv()
print(df)
