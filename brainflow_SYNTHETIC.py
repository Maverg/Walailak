import time
import numpy as np
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations

ALPHA = (8.0, 13.0)
BETA  = (13.0, 30.0)

def safe_nfft(data_len):
    nfft = DataFilter.get_nearest_power_of_two(data_len)
    while nfft >= data_len:
        nfft //= 2
    return max(64, nfft)

def welch_psd(sig, fs):
    sig = np.asarray(sig, dtype=np.float64).ravel().copy()
    DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
    nfft = safe_nfft(len(sig))
    overlap = nfft // 2
    psd, freqs = DataFilter.get_psd_welch(
        sig, nfft, overlap, fs, WindowOperations.BLACKMAN_HARRIS.value
    )
    return np.array(freqs), np.array(psd)

def band_power(freqs, psd, f_lo, f_hi):
    #Integrate PSD between f_lo and f_hi.
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    return np.trapz(psd[mask], freqs[mask])

def sliding_alpha_beta_ratio(sig, fs, win_sec=2.0, step_sec=0.5):
    #Compute time-resolved alpha/beta ratio for one channel.
    nwin = max(64, int(win_sec * fs))
    nstep = max(1, int(step_sec * fs))
    idx = np.arange(0, len(sig) - nwin + 1, nstep)
    times, ratios = [], []
    for i in idx:
        w = sig[i:i+nwin]
        f, p = welch_psd(w, fs)
        a = band_power(f, p, *ALPHA)
        b = band_power(f, p, *BETA)
        ratios.append(a / b if b > 0 else np.nan)
        times.append((i + nwin/2) / fs)
    return np.array(times), np.array(ratios)

def main(duration_sec=8, grab_window=2048):
    #acquire simulated EEG data
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)

    BoardShim.enable_board_logger()  # comment out to reduce logs

    try:
        board.prepare_session()
        fs = BoardShim.get_sampling_rate(board_id)
        eeg_ch = BoardShim.get_eeg_channels(board_id)

        # start stream and let buffer fill
        board.start_stream(45000)
        time.sleep(duration_sec)

        # grab a fixed recent window for consistency
        nwin = min(grab_window, fs * duration_sec)
        nwin = int(max(512, nwin))  # at least 512 samples
        data = board.get_current_board_data(nwin)  # shape: (channels, samples)
        board.stop_stream()
    finally:
        try:
            board.release_session()
        except Exception:
            pass

    #Extract EEG channels
    eeg_data = np.vstack([data[ch, :] for ch in eeg_ch])  # (n_ch, n_samples)

    #time-resolved α/β ratio on first EEG channel
    ch0 = eeg_data[0]
    t_ratio, ratio = sliding_alpha_beta_ratio(ch0, fs, win_sec=2.0, step_sec=0.5)

    #average PSD across channels with band shading
    #compute PSD per channel, then average the spectra
    psds, freqs = [], None
    for row in eeg_data:
        f, p = welch_psd(row, fs)
        psds.append(p)
        if freqs is None:
            freqs = f
    psd_avg = np.mean(np.vstack(psds), axis=0)

    #per-channel alpha vs beta
    alpha_p, beta_p = [], []
    for row in eeg_data:
        f, p = welch_psd(row, fs)
        alpha_p.append(band_power(f, p, *ALPHA))
        beta_p.append(band_power(f, p, *BETA))
    alpha_p = np.array(alpha_p)
    beta_p = np.array(beta_p)
    labels = [f"Ch{c}" for c in eeg_ch]

#plots
    
    #time-resolved alpha/beta ratio (simple line)
    plt.figure()
    plt.plot(t_ratio, ratio)
    plt.title("Time-resolved α/β Ratio (Channel {})".format(eeg_ch[0]))
    plt.xlabel("Time (s)")
    plt.ylabel("α/β ratio")

    #average PSD with alpha/beta shading (log power)
    plt.figure()
    # small epsilon to avoid log(0)
    plt.semilogy(freqs, psd_avg + 1e-12)
    # Shade alpha/beta ranges
    plt.axvspan(ALPHA[0], ALPHA[1], alpha=0.2, label="Alpha (8–13 Hz)")
    plt.axvspan(BETA[0],  BETA[1],  alpha=0.2, label="Beta (13–30 Hz)")
    plt.xlim(0, 45)
    plt.title("Average PSD (across EEG channels)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (a.u., log scale)")
    plt.legend()

    #per-channel alpha vs beta (grouped bars)
    x = np.arange(len(labels))
    width = 0.4
    plt.figure()
    plt.bar(x - width/2, alpha_p, width, label="Alpha (8–13 Hz)")
    plt.bar(x + width/2, beta_p,  width, label="Beta (13–30 Hz)")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Band power (a.u.)")
    plt.title("Per-channel Band Power")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(duration_sec=8, grab_window=2048)
