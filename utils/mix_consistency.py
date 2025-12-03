import torch

def enforce_mixture_consistency(est_src, mixture: torch.Tensor):
    """
    Enforce mixture consistency on separated sources.

    Handles the SepReformer codebase format where:
        est_src: list of length C (num_spks), each tensor of shape [B, T]
        mixture: [B, T]  (CMVN'ed mixture used as model input)

    We make sure that sum_c est_src[c][b, :] ≈ mixture[b, :].

    Returns est_src in the same type/structure it was given (list of tensors).
    """

    # Case 1: est_src is a list of [B, T] tensors (this is what your model uses)
    if isinstance(est_src, list):
        if len(est_src) == 0:
            return est_src

        # All sources should have same shape [B, T]
        B, T = est_src[0].shape
        device = est_src[0].device

        # Ensure mixture is [B, T] on same device
        if mixture.dim() == 3 and mixture.shape[1] == 1:
            # [B, 1, T] -> [B, T]
            mixture = mixture.squeeze(1)
        elif mixture.dim() == 1:
            # [T] -> [1, T]
            mixture = mixture.unsqueeze(0)

        if mixture.dim() != 2:
            raise RuntimeError(f"Expected mixture to be [B, T] or [B, 1, T]/[T], got {mixture.shape}")

        if mixture.shape[0] != B:
            raise RuntimeError(
                f"Mixture batch size {mixture.shape[0]} != est_src batch size {B}"
            )

        mixture = mixture.to(device)

        # Sum of estimated sources over speakers: [B, T]
        sum_est = torch.zeros_like(mixture)
        for s in est_src:
            sum_est = sum_est + s

        residual = mixture - sum_est       # [B, T]
        correction = residual / len(est_src)   # [B, T]

        # Add correction equally to each source
        corrected = [s + correction for s in est_src]
        return corrected

    # Case 2: est_src is a 3D tensor [C, B, T] (fallback/generic)
    if isinstance(est_src, torch.Tensor):
        if est_src.dim() != 3:
            raise RuntimeError(f"Expected est_src to be 3D (C,B,T), got {est_src.shape}")

        C, B, T = est_src.shape

        if mixture.dim() == 3 and mixture.shape[1] == 1:
            # [B, 1, T] -> [B, T]
            mixture = mixture.squeeze(1)
        elif mixture.dim() == 1:
            # [T] -> [1, T]
            mixture = mixture.unsqueeze(0)

        if mixture.dim() != 2:
            raise RuntimeError(f"Expected mixture to be [B, T] or [B, 1, T]/[T], got {mixture.shape}")

        if mixture.shape[0] != B:
            raise RuntimeError(
                f"Mixture batch size {mixture.shape[0]} != est_src batch size {B}"
            )

        mixture = mixture.to(est_src.device)

        sum_est = est_src.sum(dim=0)      # [B, T]
        residual = mixture - sum_est      # [B, T]
        correction = residual / C         # [B, T]
        correction = correction.unsqueeze(0)  # [1, B, T]

        return est_src + correction

    # Anything else is unexpected
    raise RuntimeError(f"Unsupported type for est_src: {type(est_src)}")
