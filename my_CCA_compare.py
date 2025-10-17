import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
import pandas as pd
from IPython.display import display


FS = 2000                
NPERSEG = 2000           
NOVERLAP = NPERSEG // 2

PATH_6R1  = r"/Users/maverick/Desktop/Frequency data/6R1"
PATH_6R2  = r"/Users/maverick/Desktop/Frequency data/6R2"
PATH_41R1 = r"/Users/maverick/Desktop/Frequency data/41R1"
PATH_41R2 = r"/Users/maverick/Desktop/Frequency data/41R2"

FULL_BAND = (2.0, 50.0)
FOCUS_6   = (5.0, 7.0)
FOCUS_41  = (40.0, 42.0)

#SNR
TARGET_6     = (5.0, 7.0)
FLANKS_6     = ((4.0, 5.0), (7.0, 8.0))
TARGET_41    = (40.0, 42.0)
FLANKS_41    = ((38.0, 40.0), (42.0, 44.0))

def _largest_numeric_2d(mat_dict):
    best, best_size = None, -1
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            size = v.shape[0] * v.shape[1]
            if size > best_size:
                best, best_size = v, size
    return best

def load_matrix(path):
    m = loadmat(path, simplify_cells=True)
    data = m.get("data")
    if not isinstance(data, np.ndarray):
        data = _largest_numeric_2d(m)
        if data is None:
            raise ValueError(f"No suitable 2D numeric array found in {path}. Keys: {list(m.keys())[:10]}")
    #samples x channels?
    if data.shape[0] < data.shape[1]:
        data = data.T
    #remove offset
    data = data - np.mean(data, axis=0, keepdims=True)
    return data

def welch_psd(x, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, pxx

def band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    # Integrate in linear units (e.g., µV^2 if data are in µV)
    return float(np.trapz(psd[mask], freqs[mask]))

def band_snr(freqs, psd, target, flanks):
    p_target = band_power(freqs, psd, target[0], target[1])
    p_f1     = band_power(freqs, psd, flanks[0][0], flanks[0][1])
    p_f2     = band_power(freqs, psd, flanks[1][0], flanks[1][1])
    p_flanks = 0.5 * (p_f1 + p_f2) + 1e-20  # avoid divide-by-zero
    snr_lin  = p_target / p_flanks
    snr_db   = 10.0 * np.log10(snr_lin)
    return snr_lin, snr_db, p_target, p_flanks

def analyze_file(mat_path, label):
    data = load_matrix(mat_path)
    if data.shape[1] < 2:
        raise ValueError(f"{label}: found only {data.shape[1]} channel(s); need at least 2.")
    ch1, ch2 = data[:, 0], data[:, 1]
    f1, pxx1 = welch_psd(ch1)
    f2, pxx2 = welch_psd(ch2)
    assert np.allclose(f1, f2), "Frequency grids differ unexpectedly."
    freqs = f1

    # 6 Hz band power & SNR
    snr6_ch1_lin, snr6_ch1_db, p6_ch1, p6_flanks_ch1 = band_snr(freqs, pxx1, TARGET_6, FLANKS_6)
    snr6_ch2_lin, snr6_ch2_db, p6_ch2, p6_flanks_ch2 = band_snr(freqs, pxx2, TARGET_6, FLANKS_6)

    # 41 Hz band power & SNR
    snr41_ch1_lin, snr41_ch1_db, p41_ch1, p41_flanks_ch1 = band_snr(freqs, pxx1, TARGET_41, FLANKS_41)
    snr41_ch2_lin, snr41_ch2_db, p41_ch2, p41_flanks_ch2 = band_snr(freqs, pxx2, TARGET_41, FLANKS_41)

    return dict(
        label=label, freqs=freqs, pxx_ch1=pxx1, pxx_ch2=pxx2,
        # 6 Hz
        p6_ch1=p6_ch1, p6_ch2=p6_ch2, p6_flanks_ch1=p6_flanks_ch1, p6_flanks_ch2=p6_flanks_ch2,
        snr6_ch1_lin=snr6_ch1_lin, snr6_ch1_db=snr6_ch1_db,
        snr6_ch2_lin=snr6_ch2_lin, snr6_ch2_db=snr6_ch2_db,
        # 41 Hz
        p41_ch1=p41_ch1, p41_ch2=p41_ch2, p41_flanks_ch1=p41_flanks_ch1, p41_flanks_ch2=p41_flanks_ch2,
        snr41_ch1_lin=snr41_ch1_lin, snr41_ch1_db=snr41_ch1_db,
        snr41_ch2_lin=snr41_ch2_lin, snr41_ch2_db=snr41_ch2_db,
    )



#outputs

def plot_full_band(a, b, title, full_band=FULL_BAND):
    f = a["freqs"]
    fb = (f >= full_band[0]) & (f <= full_band[1])

    plt.figure(figsize=(12, 6))
    plt.plot(f[fb], 10*np.log10(a["pxx_ch1"][fb]), label=f"{a['label']} ch1")
    plt.plot(f[fb], 10*np.log10(a["pxx_ch2"][fb]), label=f"{a['label']} ch2")
    plt.plot(f[fb], 10*np.log10(b["pxx_ch1"][fb]), label=f"{b['label']} ch1")
    plt.plot(f[fb], 10*np.log10(b["pxx_ch2"][fb]), label=f"{b['label']} ch2")
    plt.title(f"{title} — PSD ({full_band[0]:.0f}–{full_band[1]:.0f} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_focus(a, b, title, focus_band):
    f = a["freqs"]
    mask = (f >= focus_band[0]) & (f <= focus_band[1])

    plt.figure(figsize=(10, 5))
    plt.plot(f[mask], 10*np.log10(a["pxx_ch1"][mask]), label=f"{a['label']} ch1")
    plt.plot(f[mask], 10*np.log10(a["pxx_ch2"][mask]), label=f"{a['label']} ch2")
    plt.plot(f[mask], 10*np.log10(b["pxx_ch1"][mask]), label=f"{b['label']} ch1")
    plt.plot(f[mask], 10*np.log10(b["pxx_ch2"][mask]), label=f"{b['label']} ch2")
    plt.title(f"{title} — Focused PSD ({focus_band[0]:.1f}–{focus_band[1]:.1f} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def snr_table(title, results, which="6Hz"):
    rows = []
    for r in results:
        if which == "6Hz":
            rows.append({
                "Round": r["label"],
                #"Ch1 Target Power (5–7 Hz)": r["p6_ch1"],
                #"Ch1 Flanks Power": r["p6_flanks_ch1"],
                #"Ch1 SNR (linear)": r["snr6_ch1_lin"],
                "Ch1 SNR (dB)": r["snr6_ch1_db"],
                #"Ch2 Target Power (5–7 Hz)": r["p6_ch2"],
                #"Ch2 Flanks Power": r["p6_flanks_ch2"],
                #"Ch2 SNR (linear)": r["snr6_ch2_lin"],
                "Ch2 SNR (dB)": r["snr6_ch2_db"],
            })
        else:
            rows.append({n
                "Round": r["label"],
                #"Ch1 Target Power (40–42 Hz)": r["p41_ch1"],
                #"Ch1 Flanks Power": r["p41_flanks_ch1"],
                #"Ch1 SNR (linear)": r["snr41_ch1_lin"],
                "Ch1 SNR (dB)": r["snr41_ch1_db"],
                #"Ch2 Target Power (40–42 Hz)": r["p41_ch2"],
                #"Ch2 Flanks Power": r["p41_flanks_ch2"],
                #"Ch2 SNR (linear)": r["snr41_ch2_lin"],
                "Ch2 SNR (dB)": r["snr41_ch2_db"],
            })
    df = pd.DataFrame(rows)
    print(f"\n=== {title} — {which} SNR ===")
    display(df.round(4))

six_r1   = analyze_file(PATH_6R1,  "6R1")
six_r2   = analyze_file(PATH_6R2,  "6R2")
forty_r1 = analyze_file(PATH_41R1, "41R1")
forty_r2 = analyze_file(PATH_41R2, "41R2")

#plots - 6hz
plot_full_band(six_r1, six_r2, title="6 Hz — Round 1 vs Round 2")
plot_focus(six_r1, six_r2, title="6 Hz — Round 1 vs Round 2", focus_band=FOCUS_6)

#plots - 41hz
plot_full_band(forty_r1, forty_r2, title="41 Hz — Round 1 vs Round 2")
plot_focus(forty_r1, forty_r2, title="41 Hz — Round 1 vs Round 2", focus_band=FOCUS_41)

snr_table("6 Hz Pair",   [six_r1, six_r2], which="6Hz")
snr_table("41 Hz Pair",  [forty_r1, forty_r2], which="41Hz")
