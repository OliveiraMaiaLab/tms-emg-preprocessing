# MEP Preprocessing

Offline analysis of MEPs was performed using a structured preprocessing pipeline. Raw EMG signals were denoised with a three-level adaptive wavelet filter (Daubechies family, db1) and BayesShrink soft thresholding (Chang et al., 2000). Wavelet filtering is particularly effective for non-stationary signals and for separating noise sources with overlapping frequency content (Samann & Schanze, 2019; Wahab & O’Haver, 2020; Machetanz et al., 2021). A third-order Savitzky–Golay filter with a window size of 5 (Awal et al., 2011) was then applied for smoothing. This approach was preferred over conventional band-pass filtering, which introduced distortions in our dataset.

Following preprocessing, MEPs were screened according to established exclusion criteria. Trials were discarded if they showed pre-activation or if peak-to-peak amplitude was <50 μV. Background EMG was assessed with an automated procedure: a 500 ms artifact-free segment of EMG was selected as baseline, and the 50 ms preceding each TMS pulse were compared against it. Pre-activation was defined when the rectified mean amplitude exceeded baseline by >2 SD (McCambridge et al., 2020) or if the root mean square amplitude exceeded 15 μV (Hinder et al., 2014). Trials meeting both conditions were excluded.

The pipeline is designed for MEP analysis and amplitude extraction, requires minimal coding expertise, and is available at `YYY@github.com` alongside a step-by-step guide.

---

<details>
<summary><strong>4.1. File Selection for Preprocessing</strong></summary>

Set the `name` variable to the filename and run the section. Filenames must follow the structure:

{subID}{ses}{hemi}_yyyy-mm-dd_HH-MM-SS

![Alt text](images\startup_menu.png)


</details>

<details>
<summary><strong>4.2. EMG Segmentation</strong></summary>

After running the section, the EMG trace and TMS pulse time series will be displayed. Segmentation of the signal begins here.

<details>
<summary><strong>4.2.1. First-Level Segmentation</strong></summary>

Split the EMG signal into segments corresponding to different phases of the experiment.

**Figure layout:**

- **Left panel (top):** TMS pulse markers  
- **Left panel (bottom):** Raw EMG signal  
- **Right panel:** Code snippet to be edited  

**To segment manually:**

1. Use the figure cursor in Spyder (left panel) to locate the end of each segment, using TMS pulse markers as reference.  
2. Enter the values into the corresponding segment variables in the script (right panel).  
3. Define the reference EMG used for baseline control by choosing the start of a 500 ms window with stable EMG activity and enter it in `ref_emg`.  

After filling in the variables, execute and advance to the next section.

</details>

<details>
<summary><strong>4.2.2. MVIC Segmentation</strong></summary>

Define the start of a 3 s window for MVIC epochs. This step can be skipped if MEP amplitude is analyzed without normalization.

**Figure layout:**

- **Left panel:** MVIC raw trace  
- **Right panel (top):** Code snippet  
- **Right panel (bottom):** MVIC epoch plot  

Run the section to display the selected MVIC epoch. The amplitude should be stable across the window. If adjustments are needed, update the variable and rerun the section until satisfactory. Then continue to the next section.

</details>

<details>
<summary><strong>4.2.3. MEP Epoch Definition</strong></summary>

All MEPs will be plotted together. Define the analysis window by specifying the time range (in ms) relative to the TMS pulse.

**Examples of MEP overlap plots:**

- Without TMS artifact (top left)  
- With large artifact (top right)  
- With complex morphology (bottom left)  
- Bottom right: Code snippet  

</details>

</details>

<details>
<summary><strong>4.3. Visual Inspection of Automatic Peak Detection</strong></summary>

MEPs from each single-pulse block are plotted with markers around detected peaks. Inspect the plots and list in the code snippet the MEP IDs where automatic detection failed.

**Figure layout:**

- **Top:** Example of MEPs with automatic markers. In Pulse 25 the maximum peak requires manual correction.  
- **Bottom:** Code snippet to be edited  

</details>

<details>
<summary><strong>4.4. Correction of Flagged MEPs</strong></summary>

Flagged MEPs are plotted in detail. Use the cursor to identify the correct peak values and record them in the generated Excel file.

**Figure layout:**

- **Left:** Detailed view of a flagged MEP (Pulse 25), with red cross indicating the correct peak  
- **Right:** Excel file for manual correction of peak values

</details>


