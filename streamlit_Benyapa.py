import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from scipy.signal import welch
import mne

FILES = ["data/Tab1_01.mat", "data/Tab1_02.mat", "data/Tab1_03.mat"]


BANDS = {
    "Delta (1–4 Hz)": (1, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–45 Hz)": (30, 45),
}


def mat_to_py(obj):
    #convert mat to dict recursively
    if hasattr(obj, "_fieldnames"):
        return {field: mat_to_py(getattr(obj, field)) for field in obj._fieldnames}
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [mat_to_py(o) for o in obj.ravel()]
        return obj
    else:
        return obj

def extract_alleeg(filepath):
    mat = sio.loadmat(filepath, squeeze_me=False, struct_as_record=False)
    alleeg = mat.get("ALLEEG", None)
    if alleeg is None:
        raise ValueError(f"No ALLEEG variable found in {filepath}")
    return [mat_to_py(alleeg[0, i]) for i in range(alleeg.shape[1])]

def compute_band_powers(data, srate, freqs, psds):
    band_dict = {}
    for band_name, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bp = np.trapz(psds[:, idx], freqs[idx], axis=1)
        band_dict[band_name] = bp
    return band_dict

@st.cache_data
def load_all_datasets():
    datasets_all = []
    for file in FILES:
        ds_list = extract_alleeg(file)
        for ds in ds_list:
            data = np.array(ds["data"], dtype=float)
            srate = float(np.squeeze(ds["srate"]))
            chanlocs = ds.get("chanlocs", None)
            if chanlocs:
                labels = []
                for ch in chanlocs:
                    if isinstance(ch, dict) and "labels" in ch:
                        lab = str(np.squeeze(ch["labels"]))
                    else:
                        lab = f"Ch{len(labels)+1}"
                    labels.append(lab)
            else:
                labels = [f"Ch{i+1}" for i in range(data.shape[0])]
            datasets_all.append({
                "data": data,
                "srate": srate,
                "labels": labels
            })
    return datasets_all

datasets = load_all_datasets()

# Dashboard
st.title("Dashboard")

dataset_idx = st.sidebar.slider("Select Dataset", 0, len(datasets)-1, 0)
ds = datasets[dataset_idx]
data, srate, labels = ds["data"], ds["srate"], ds["labels"]

st.write(f"### Dataset {dataset_idx+1}")
st.write(f"- Channels: {len(labels)}")
st.write(f"- Sampling Rate: {srate} Hz")
st.write(f"- Duration: {data.shape[1]/srate:.2f} seconds")

# PSD
n_channels, n_times = data.shape
nperseg = min(4096, max(256, 2 ** int(np.floor(np.log2(n_times // 8)))))
noverlap = nperseg // 2

freqs = None
psds = []
for ch in range(n_channels):
    f, pxx = welch(data[ch, :], fs=srate, nperseg=nperseg, noverlap=noverlap)
    if freqs is None:
        freqs = f
    psds.append(pxx)
psds = np.vstack(psds)

# Band Power
band_dict = compute_band_powers(data, srate, freqs, psds)
band_df = pd.DataFrame({"Channel": labels, **band_dict})
st.dataframe(band_df.round(4))



# plots

#PSD
st.subheader("Channel PSDs (Power Spectral Density)")
selected_channels = st.multiselect("Select Channels to Plot", labels, default=labels[:8])
fig, ax = plt.subplots(figsize=(9, 5))
for ch_name in selected_channels:
    idx = labels.index(ch_name)
    ax.plot(freqs, 10*np.log10(psds[idx, :] + 1e-20), label=ch_name)
ax.set_xlim(0, 50)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (dB/Hz)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Band Power
st.subheader("Average Band Power Across Channels")
avg_band = [np.mean(band_df[band]) for band in BANDS.keys()]
fig_bar = px.bar(
    x=list(BANDS.keys()),
    y=avg_band,
    labels={"x": "Frequency Band", "y": "Average Power"},
    title="Mean Band Power (µV²/Hz)"
)
st.plotly_chart(fig_bar, use_container_width=True)




from sklearn.cross_decomposition import CCA

st.subheader("CCA")

# pick channel groups
st.markdown("Select two groups of channels to analyze canonical correlation between them.")

group1 = st.multiselect("Group 1 Channels", labels, default=labels[:8])
group2 = st.multiselect("Group 2 Channels", labels, default=labels[8:16])

if group1 and group2:
    # Extract signal matrices
    X = np.array([data[labels.index(ch), :] for ch in group1]).T
    Y = np.array([data[labels.index(ch), :] for ch in group2]).T

    # Run CCA
    n_components = st.slider("Number of CCA Components", 1, min(X.shape[1], Y.shape[1]), 1)
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    U, V = cca.transform(X, Y)

    # Compute CCA
    canonical_corrs = [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(n_components)]

    st.write(f"**Canonical Correlation Coefficients:** {np.round(canonical_corrs, 3)}")
    st.write(f"Highest canonical correlation: **{np.max(canonical_corrs):.3f}**")

    # plort
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(U[:, 0], label="U₁ (Group 1)")
    ax.plot(V[:, 0], label="V₁ (Group 2)")
    ax.set_title("First Canonical Variate Pair (Time Series)")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend()
    st.pyplot(fig)

    # Optional scatter plot of first canonical pair
    fig2 = px.scatter(
        x=U[:, 0], y=V[:, 0],
        labels={"x": "U₁ (Group 1)", "y": "V₁ (Group 2)"},
        title="Canonical Variate Relationship (First Component)",
        trendline="ols"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Please select at least one channel in each group to compute CCA.")

