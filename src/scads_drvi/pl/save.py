"""Writing a figure out, once.

The idiom this replaces was typed at every figure -- a PDF at 300 dpi and a PNG at 200,
both with ``bbox_inches="tight"`` -- roughly twenty times across the notebooks, with one
notebook wrapping it locally and the rest not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = ["SaveSpec", "save_figure"]


@dataclass(frozen=True)
class SaveSpec:
    """Formats and resolutions for one figure export."""

    formats: tuple[str, ...] = ("pdf", "png")
    dpi: Mapping[str, int] = field(
        default_factory=lambda: {"pdf": 300, "png": 200}
    )
    bbox_inches: str | None = "tight"
    close: bool = False
    transparent: bool = False

    def dpi_for(self, fmt: str) -> int:
        return int(self.dpi.get(fmt, 300))


def save_figure(
    fig: Figure,
    name: str,
    outdir: str | Path,
    *,
    spec: SaveSpec | None = None,
    metadata: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write `fig` to every format in `spec` under `outdir`; return the paths written.

    `name` carries no extension -- the formats decide that. `outdir` is created if
    needed.

    `metadata` is embedded where the format supports it, so a stray PDF can be traced
    back to the run that made it. That is cheap here and the alternative is a figure
    nobody can attribute.
    """
    spec = spec or SaveSpec()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if Path(name).suffix.lstrip(".") in spec.formats:
        raise ValueError(
            f"name should carry no extension; got {name!r} and formats "
            f"{spec.formats}. The formats decide the suffix."
        )

    written: list[Path] = []
    for fmt in spec.formats:
        path = outdir / f"{name}.{fmt}"
        kwargs: dict = {
            "dpi": spec.dpi_for(fmt),
            "bbox_inches": spec.bbox_inches,
            "transparent": spec.transparent,
        }
        if metadata:
            kwargs["metadata"] = _format_metadata(fmt, metadata)
        fig.savefig(path, **kwargs)
        written.append(path)

    if spec.close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def _format_metadata(fmt: str, metadata: Mapping[str, str]) -> dict:
    """Map free-form provenance onto the keys each backend accepts.

    The PDF and PNG backends take different key sets and raise on unknown ones, so a
    single dict handed to both is a silent portability trap.
    """
    flat = "; ".join(f"{k}={v}" for k, v in metadata.items())
    if fmt == "pdf":
        return {"Keywords": flat}
    if fmt == "png":
        return {"Description": flat}
    if fmt == "svg":
        return {"Description": flat}
    return {}


def figure_metadata(**fields: str) -> dict[str, str]:
    """Provenance fields for :func:`save_figure`, with a UTC timestamp added."""
    from datetime import datetime, timezone

    out = {k: str(v) for k, v in fields.items() if v is not None}
    out.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return out


__all__.append("figure_metadata")
