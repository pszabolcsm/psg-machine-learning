import os
import numpy as np
import wfdb
import pyedflib
import pandas as pd
from pyPSG.biomarkers.get_ecg_bm import get_ecg_biomarkers
from pyPSG.biomarkers.get_ppg_bm import get_ppg_biomarkers
from pyPSG.IO.edf_read import read_edf_signals

def wfdb_to_edf(record_name, output_path, data_dir=""):
    """
       Convert a WFDB record to an EDF+ file.

       Parameters
       ----------
       record_name : str
           Name (or base name) of the WFDB record to be read. This is typically the
           file name without extension, e.g. "record01".
       output_path : str
           Full path (including file name) where the EDF+ file will be written.
       data_dir : str, optional
           Directory that contains the WFDB record files. If empty, the current
           working directory is used.

       Returns
       -------
       None
           The function writes an EDF+ file to `output_path` and does not return
           a value.
       """
    # Build full path to the WFDB record (without extension)
    record_path = os.path.join(data_dir, record_name)
    
    # Read the WFDB record in physical units
    rec = wfdb.rdrecord(record_path, physical=True)

    # Get the Sampling frequency, number of signals, signal names, physical units and signal data
    fs = float(rec.fs)
    n_sig = int(rec.n_sig)
    names = list(rec.sig_name)
    units = list(rec.units) if hasattr(rec, "units") else [""] * n_sig
    data = rec.p_signal
    
    # Containers for EDF writer
    all_sig = []
    channel_info = []
    failed_sig = 0 # number of channels that are invalid and skipped
    
    # Build EDF channel headers and filter out unusable channels.
    for i in range(n_sig):
        # Extract a single channel
        s = data[:, i].astype(float)
        # Compute physical min and max
        pmin = float(np.nanmin(s))
        pmax = float(np.nanmax(s))
        # If min or max is NaN, this channel is effectively invalid
        if np.isnan(pmin) or np.isnan(pmax):
            failed_sig += 1
            continue
        if pmax == pmin:
            pmax += 0.001
        
        # Create the EDF channel header dictionary
        channel_info.append({
            "label": names[i],
            "dimension": units[i],
            "sample_frequency": fs,
            "physical_min": pmin,
            "physical_max": pmax,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        })
        # Store the channel data to be written
        all_sig.append(s)
    # Write the EDF+ file
    with pyedflib.EdfWriter(output_path, (n_sig-failed_sig), file_type=pyedflib.FILETYPE_EDFPLUS) as ew:
        ew.setSignalHeaders(channel_info)
        ew.writeSamples(all_sig)

def is_flat_or_zero(sig, zero_ratio=0.99, min_std=1e-12):
    s = np.asarray(sig)
    if s.size == 0 or np.all(np.isnan(s)):
        return True
    # every sample 0
    if np.nanmax(np.abs(s)) == 0 or (np.count_nonzero(s == 0) / s.size) >= zero_ratio:
        return True
    # Small std = constant signal
    if np.nanstd(s) < min_std:
        return True
    return False

def _asdict(x):
    return x if isinstance(x, dict) else {}

def split_into_windows(signal, fs, window_sec=30.0, overlap_sec=0.0, drop_last=True):
    s = np.asarray(signal)
    n = len(s)
    if n == 0 or np.isnan(s).all():
        return []
    w = int(round(window_sec * fs))
    step = int(round((window_sec - overlap_sec) * fs))
    if w <= 0 or step <= 0:
        return []

    idx_pairs = []
    start = 0
    while start < n:
        end = start + w
        if end <= n:
            idx_pairs.append((start, end))
        else:
            if not drop_last:
                idx_pairs.append((start, n))
            break
        start += step
    return idx_pairs


if __name__ == "__main__":
    directory = 'training'  # set directory path
    matlab_path = '/home/shared/software/matlab/R2024b/'
    
    # Iterate over all WFDB header files in the directory and convert them to EDF+
    for entry in os.scandir(directory):
        if entry.is_file():  # check if it's a file
            if entry.name.endswith('.hea'): # chech if it's a wfdb file
                name = entry.name.replace('.hea', '') # base record name without extension
                print(name)
                wfdb_to_edf(name, f"EDFs/{name}.edf", directory)
    # -- CONFIGURATIONS --
    window_sec = 30.0
    overlap_sec = 0.0
    drop_last = False
    
    # Path to the CSV file
    csv_path = "biomarkers/all_bms_mean_windowed.csv"
    
    # If the CSV already exists, load previously computed biomarkers
    if os.path.exists(csv_path):
        all_bms = pd.read_csv(csv_path)
        # Number of unique patients already present in the aggregated biomarker table
        aa = all_bms['Patient_id'].nunique()
        if "Patient_id" in all_bms.columns:
            all_bms["Patient_id"] = all_bms["Patient_id"].astype(str)
            # Build a set of already processed (patient, window) pairs
        processed = set(zip(all_bms.get("Patient_id", pd.Series(dtype=str)).astype(str),
                            all_bms.get("Window_idx", pd.Series(dtype=int)).astype(int, errors="ignore")))
    # If no previous CSV exists, start with an empty DataFrame where biomarkers will be stored
    else:
        all_bms = pd.DataFrame()
        aa = 0
        processed = set()
    
    # Make sure the output directory for biomarkers exists
    os.makedirs("biomarkers", exist_ok=True)
    
    # Iterate through all files in the EDFs directory
    for entry in os.scandir("EDFs"):
        if not (entry.is_file() and entry.name.endswith('.edf')):# check if it's an EDF file
            continue
        # Extract patient ID from EDF filename
        pid = entry.name[:-4]
        # Read selected signals
        signals = read_edf_signals(f"EDFs/{entry.name}", ["PLETH", "II", "V"])
        # Dictionary to store windowing information for each channel
        channel_windows = {}
        # Track the maximum number of windows across channels
        max_windows = 0
        # If this patient is not yet in the biomarkers table, update the counter
        if pid not in set(all_bms['Patient_id'].unique()):
            aa += 1
        per = (aa / 750) * 100
        
        # Process each channel for window splitting
        for ch in ["II", "V", "PLETH"]:
            if ch in signals:
                sig = signals[ch]["signal"]
                fs = signals[ch]["fs"]
                # Split the signal into time windows according to configuration
                win_idx = split_into_windows(sig, fs, window_sec=window_sec,
                                             overlap_sec=overlap_sec, drop_last=drop_last)
                # Store sampling frequency, window indices and raw signal for this channel
                channel_windows[ch] = {"fs": fs, "wins": win_idx, "sig": sig}
                # Update maximum number of windows observed among all channels
                max_windows = max(max_windows, len(win_idx))
        
        # Skip this record if no windows were created for any channel
        if max_windows == 0:
            continue
        
        # Iterate over all possible window indices for this patient
        for w_idx in range(max_windows):
            # Skip windows that have already been processed earlier
            if (str(pid), int(w_idx)) in processed:
                continue
            # Initialize a dictionary to store biomarkers for the current window
            row_dict = {"Patient_id": pid, "Window_idx": int(w_idx)}
            print(entry.name)
            print(f"{per:.3f}%, {aa}. elem, {w_idx + 1}. ablak")
            
            # Process lead II (first ECG channel) if available and the window index is valid
            if "II" in channel_windows and w_idx < len(channel_windows["II"]["wins"]):
                fs = channel_windows["II"]["fs"]
                sidx, eidx = channel_windows["II"]["wins"][w_idx]
                win_sig = channel_windows["II"]["sig"][sidx:eidx]
                # Store the window start time (in seconds) for this row, if not already set
                row_dict["Window_start_s"] = row_dict.get("Window_start_s", sidx / fs)
                
                # Skip flat or zero signal segments
                if not is_flat_or_zero(win_sig):
                    # Extract ECG and HRV biomarkers for this window from lead II
                    ecg1_biomarkers = get_ecg_biomarkers(win_sig, fs, matlab_path, get_hrv=True)
                    
                    # Merge ECG statistics
                    stats = (_asdict(ecg1_biomarkers["ecg"].get("stat_i", {})) |
                             _asdict(ecg1_biomarkers["ecg"].get("stat_w", {})))
                    
                    # Add mean ECG biomarkers to the row with "ECG1_" prefix
                    for name, values in stats.items():
                        if "mean" in values:
                            row_dict["ECG1_" + name] = values["mean"]
                            
                    # Add mean HRV biomarkers to the row with "HRV1_" prefix
                    for name, values in ecg1_biomarkers['hrv']['stats'].items():
                        if "mean" in values:
                            row_dict["HRV1_" + name] = values["mean"]
            
            # Process lead V (second ECG channel) if available and the window index is valid
            if "V" in channel_windows and w_idx < len(channel_windows["V"]["wins"]):
                fs = channel_windows["V"]["fs"]
                sidx, eidx = channel_windows["V"]["wins"][w_idx]
                win_sig = channel_windows["V"]["sig"][sidx:eidx]
                # Store the window start time (in seconds) for this row, if not already set
                row_dict.setdefault("Window_start_s", sidx / fs)
                
                # Skip flat or zero signal segments
                if not is_flat_or_zero(win_sig):
                    # Extract ECG and HRV biomarkers for this window from lead II
                    ecg2_biomarkers = get_ecg_biomarkers(win_sig, fs, matlab_path, get_hrv=True)
                    
                    # Merge ECG statistics
                    stats = (_asdict(ecg2_biomarkers["ecg"].get("stat_i", {})) |
                             _asdict(ecg2_biomarkers["ecg"].get("stat_w", {})))
                    
                    # Add mean ECG biomarkers to the row with "ECG2_" prefix
                    for name, values in stats.items():
                        if "mean" in values:
                            row_dict["ECG2_" + name] = values["mean"]
                    # Add mean HRV biomarkers to the row with "HRV2_" prefix
                    for name, values in ecg2_biomarkers['hrv']['stats'].items():
                        if "mean" in values:
                            row_dict["HRV2_" + name] = values["mean"]
            
            # Process PPG channel if available and the window index is valid
            if "PLETH" in channel_windows and w_idx < len(channel_windows["PLETH"]["wins"]):
                fs = channel_windows["PLETH"]["fs"]
                sidx, eidx = channel_windows["PLETH"]["wins"][w_idx]
                win_sig = channel_windows["PLETH"]["sig"][sidx:eidx]
                # Store the window start time (in seconds) for this row, if not already set
                row_dict.setdefault("Window_start_s", sidx / fs)
                
                # Skip flat or zero signal segments
                if not is_flat_or_zero(win_sig):
                    # Extract PPG and BRV biomarkers for this window from PPG
                    ppg_biomarkers = get_ppg_biomarkers(win_sig, fs, get_brv=True)
                    
                    # Add mean PPG biomarkers to the row with "PPG_" prefix
                    for cat, block in ppg_biomarkers["ppg"].bm_stats.items():
                        for name, values in block.items():
                            if "mean" in values:
                                row_dict["PPG_" + name] = values["mean"]
                    # Add mean BRV biomarkers to the row with "BRV_" prefix
                    for name, values in ppg_biomarkers['brv']['stats'].items():
                        if "mean" in values:
                            row_dict["BRV_" + name] = values["mean"]
            
            # Append the current window's biomarkers as a new row to the main DataFrame and save it
            all_bms = pd.concat([all_bms, pd.DataFrame([row_dict])], ignore_index=True)
            all_bms.to_csv(csv_path, index=False)
            # Mark this patient's window as processed
            processed.add((str(pid), int(w_idx)))