import torch
import torch.nn.functional as F

def _stft(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """
    x: [B, T]
    returns complex STFT: [B, F, T']
    """
    window = torch.hann_window(win_length, device=x.device)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    return X

def multi_res_stft_loss(
    est_src: torch.Tensor,
    ref_src: torch.Tensor,
    fft_sizes=(256, 512, 1024),
    hop_sizes=(64, 128, 256),
    win_lengths=(256, 512, 1024),
) -> torch.Tensor:
    """
    Multi-resolution STFT loss between estimated and reference sources.

    est_src: [C, B, T] or [B, C, T]
    ref_src: same shape as est_src (or broadcastable)

    We flatten speaker/batch into one dimension and apply STFT loss per waveform,
    then average.
    """

    # Make sure both are on same device
    ref_src = ref_src.to(est_src.device)

    # Align shapes: we'll just flatten everything except last dim
    if est_src.dim() != 3:
        raise RuntimeError(f"Expected est_src to be 3D (C,B,T or B,C,T), got {est_src.shape}")
    if ref_src.dim() != 3:
        raise RuntimeError(f"Expected ref_src to be 3D, got {ref_src.shape}")

    # reshape to [N, T]
    est_flat = est_src.reshape(-1, est_src.shape[-1])  # [N, T]
    ref_flat = ref_src.reshape(-1, ref_src.shape[-1])  # [N, T]

    total_loss = 0.0
    num_resolutions = len(fft_sizes)

    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        est_stft = _stft(est_flat, n_fft, hop, win)  # [N, F, T']
        ref_stft = _stft(ref_flat, n_fft, hop, win)

        est_mag = torch.abs(est_stft)
        ref_mag = torch.abs(ref_stft)

        # Magnitude L1
        mag_loss = F.l1_loss(est_mag, ref_mag)

        # Log-magnitude L1
        est_log = torch.log(est_mag + 1e-7)
        ref_log = torch.log(ref_mag + 1e-7)
        logmag_loss = F.l1_loss(est_log, ref_log)

        total_loss = total_loss + mag_loss + logmag_loss

    return total_loss / num_resolutions
