# Complete Delay Discounting Pipeline
## Delay Discounting
The adopi.py code on running starts the psychopy delay discounting code which is saved in a new data folder automatically created. You can change/rename the file and bring it over to data_files for it to be considered into the final plot.
## Band Values
EEG data stored in edf file, should have equivalent name before the _EPOCFLEX_ for it to be mapped to appropraite Delay Discounting file of the same participant.
## Preprocessing and Value Calculation
the preprocessing of teh eeg file is done by the calculate_eeg_value.py file which also calculates the average of 4 band frequency range and outputs the values.
## Plotting
Finally the eeg_value.csv which contains the output of each edf file processed, on running add_k_values.py adds the last mean_k value which is the input to plotter.py, which plots discounting rates on y axis and eeg_values on the y- axis to give the final graph.
