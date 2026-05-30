import argparse
from typing import Any, Dict, Optional, Tuple, Union


FLUX2_T3_FACTOR_PATTERN_4K = ((1, 1), (8, 16), (16, 8), (4, 32), (32, 4))
FLUX2_T3_FACTOR_PATTERN_8K = ((1, 1), (16, 32), (32, 16), (8, 64), (64, 8))


def parse_flux2_window_size(value: Union[str, int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"Flux2 local attention window size must be positive, but got {value}.")
        return (value, value)

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Flux2 local attention window size must have two elements, but got {value}.")
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Flux2 local attention window size must be positive, but got {value}.")
        return (height, width)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported Flux2 local attention window size type: {type(value)!r}")

    text = value.strip().lower().replace("*", "x").replace(",", "x")
    if text == "":
        raise ValueError("Flux2 local attention window size cannot be empty.")
    if "x" not in text:
        size = int(text)
        if size <= 0:
            raise ValueError(f"Flux2 local attention window size must be positive, but got {value}.")
        return (size, size)

    height_str, width_str = text.split("x", 1)
    height, width = int(height_str), int(width_str)
    if height <= 0 or width <= 0:
        raise ValueError(f"Flux2 local attention window size must be positive, but got {value}.")
    return (height, width)


def parse_flux2_local_factor_pattern(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none", "off", "false", "0"):
            return None
        if text in ("auto", "t3-auto", "t3video-auto", "t3-video-auto"):
            return "auto"
        if text in ("t3-4k", "t3_4k", "fixed-4k", "4k"):
            return FLUX2_T3_FACTOR_PATTERN_4K
        if text in ("t3-8k", "t3_8k", "t3-10k", "t3_10k", "fixed-8k", "fixed-10k", "8k", "10k"):
            return FLUX2_T3_FACTOR_PATTERN_8K

        pattern = []
        for item in text.replace(";", ",").split(","):
            item = item.strip().replace("*", "x")
            if item == "":
                continue
            if "x" in item:
                height_str, width_str = item.split("x", 1)
                height, width = int(height_str), int(width_str)
            else:
                height = width = int(item)
            if height <= 0 or width <= 0:
                raise ValueError(f"Flux2 local factor pattern entries must be positive, but got {item!r}.")
            pattern.append((height, width))
        if len(pattern) == 0:
            return None
        return tuple(pattern)

    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, int) for item in value):
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError(f"Flux2 local factor pattern entries must be positive, but got {value}.")
        return (tuple(value),)

    pattern = []
    for item in value:
        if isinstance(item, int):
            height = width = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            height, width = int(item[0]), int(item[1])
        else:
            raise TypeError(f"Unsupported Flux2 local factor pattern entry: {item!r}")
        if height <= 0 or width <= 0:
            raise ValueError(f"Flux2 local factor pattern entries must be positive, but got {item!r}.")
        pattern.append((height, width))
    return tuple(pattern) if len(pattern) > 0 else None


def add_flux2_local_attention_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--flux2_local_attention",
        action="store_true",
        help="Enable Flux2 close+remote local image attention on the non-USP path.",
    )
    parser.add_argument(
        "--flux2_window_size",
        type=str,
        default="16",
        help=(
            "Flux2 local attention close partition size on the latent-token grid "
            "(not pixels), as HxW or a single integer. One latent token corresponds "
            "to a 16x16 image region."
        ),
    )
    parser.add_argument(
        "--flux2_local_max_windows_per_batch",
        type=int,
        default=16,
        help=(
            "Max number of close/remote partitions to batch into one attention call. "
            "Use a small positive value to cap peak memory. `-1` processes all windows "
            "in each same-shaped bucket at once, which can erase most memory savings."
        ),
    )
    parser.add_argument(
        "--flux2_local_factor_pattern",
        type=str,
        default=None,
        help=(
            "T3-Video-style per-layer local attention factors, e.g. `8x8,8x16,16x8`. "
            "Each HxW entry means number of close windows along latent height/width; "
            "the actual close window is latent_size / factor, and the remote branch "
            "uses the factor as its stride. Use `auto` to derive a 5-entry pattern "
            "from the current latent size and `--flux2_window_size`, or `t3-4k`/`t3-8k` "
            "for fixed T3-style presets."
        ),
    )
    parser.add_argument(
        "--flux2_single_stream_seq_chunk_size",
        type=int,
        default=0,
        help=(
            "Sequence chunk size for Flux2 single-stream blocks. The fused QKV/MLP input "
            "projection and output projection are processed chunk-by-chunk to reduce peak "
            "activation memory. `0` disables chunking."
        ),
    )
    parser.add_argument(
        "--flux2_double_stream_seq_chunk_size",
        type=int,
        default=None,
        help=(
            "Sequence chunk size for Flux2 double-stream image QKV/output projections "
            "and image FFN. Defaults to `--flux2_single_stream_seq_chunk_size` when "
            "omitted. Use `0` to disable double-stream chunking."
        ),
    )
    return parser


def build_flux2_joint_attention_kwargs(
    args: Optional[argparse.Namespace] = None,
    *,
    enabled: Optional[bool] = None,
    window_size: Optional[Union[str, int, Tuple[int, int]]] = None,
    max_windows_per_batch: Optional[int] = None,
    factor_pattern: Optional[Any] = None,
    single_stream_seq_chunk_size: Optional[int] = None,
    double_stream_seq_chunk_size: Optional[int] = None,
    use_usp: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    if args is not None:
        if enabled is None:
            enabled = bool(getattr(args, "flux2_local_attention", False))
        if window_size is None:
            window_size = getattr(args, "flux2_window_size", "16")
        if max_windows_per_batch is None:
            max_windows_per_batch = getattr(args, "flux2_local_max_windows_per_batch", 16)
        if factor_pattern is None:
            factor_pattern = getattr(args, "flux2_local_factor_pattern", None)
        if single_stream_seq_chunk_size is None:
            single_stream_seq_chunk_size = getattr(args, "flux2_single_stream_seq_chunk_size", 0)
        if double_stream_seq_chunk_size is None:
            double_stream_seq_chunk_size = getattr(args, "flux2_double_stream_seq_chunk_size", None)
        if use_usp is None:
            use_usp = bool(getattr(args, "use_usp", False))

    enabled = bool(enabled)
    double_stream_seq_chunk_size_was_none = double_stream_seq_chunk_size is None
    single_stream_seq_chunk_size = 0 if single_stream_seq_chunk_size is None else int(single_stream_seq_chunk_size)
    if single_stream_seq_chunk_size < 0:
        raise ValueError(
            "`--flux2_single_stream_seq_chunk_size` must be a non-negative integer, "
            f"but got {single_stream_seq_chunk_size}."
        )
    if double_stream_seq_chunk_size_was_none:
        double_stream_seq_chunk_size = single_stream_seq_chunk_size
    else:
        double_stream_seq_chunk_size = int(double_stream_seq_chunk_size)
    if double_stream_seq_chunk_size < 0:
        raise ValueError(
            "`--flux2_double_stream_seq_chunk_size` must be a non-negative integer, "
            f"but got {double_stream_seq_chunk_size}."
        )

    if not enabled and single_stream_seq_chunk_size == 0 and double_stream_seq_chunk_size == 0:
        return None

    if use_usp:
        if enabled:
            raise ValueError(
                "Flux2 local attention currently only supports the non-USP path. "
                "Disable `--use_usp` when enabling `--flux2_local_attention`."
            )
        if single_stream_seq_chunk_size > 0:
            raise ValueError(
                "Flux2 single-stream sequence chunking currently only supports the non-USP path. "
                "Disable `--use_usp` when enabling `--flux2_single_stream_seq_chunk_size`."
            )
        if double_stream_seq_chunk_size > 0:
            raise ValueError(
                "Flux2 double-stream sequence chunking currently only supports the non-USP path. "
                "Disable `--use_usp` when enabling `--flux2_double_stream_seq_chunk_size`."
            )

    kwargs = {}
    if enabled:
        window_size = parse_flux2_window_size("16" if window_size is None else window_size)
        factor_pattern = parse_flux2_local_factor_pattern(factor_pattern)
        max_windows_per_batch = 16 if max_windows_per_batch is None else int(max_windows_per_batch)
        if max_windows_per_batch == -1:
            pass
        elif max_windows_per_batch <= 0:
            raise ValueError(
                "`--flux2_local_max_windows_per_batch` must be a positive integer or -1, "
                f"but got {max_windows_per_batch}."
            )

        kwargs.update(
            {
                "flux2_local_attention": True,
                "flux2_window_size": window_size,
                "flux2_local_max_windows_per_batch": max_windows_per_batch,
            }
        )
        if factor_pattern is not None:
            kwargs["flux2_local_factor_pattern"] = factor_pattern

    if single_stream_seq_chunk_size > 0:
        kwargs["flux2_single_stream_seq_chunk_size"] = single_stream_seq_chunk_size
    if double_stream_seq_chunk_size > 0:
        kwargs["flux2_double_stream_seq_chunk_size"] = double_stream_seq_chunk_size
    elif not double_stream_seq_chunk_size_was_none:
        kwargs["flux2_double_stream_seq_chunk_size"] = 0

    return kwargs or None
