from __future__ import annotations

from .schemas import Provenance


def synthetic(label: str, **components: str) -> Provenance:
    return Provenance(
        source="synthetic",
        label=label,
        components=components,
        notes=["Programmatically generated data or artifacts; not measured from a real vehicle run."],
    )


def mock(label: str, **components: str) -> Provenance:
    return Provenance(
        source="mock",
        label=label,
        components=components,
        notes=["Demo placeholder path used to keep the UI loop runnable; do not report as an experimental result."],
    )


def toy_env(label: str, **components: str) -> Provenance:
    return Provenance(
        source="toy_env",
        label=label,
        components=components,
        notes=["Toy environment result for interface validation, not a physical off-road benchmark."],
    )


def real_data(label: str, data_sources: list[str] | None = None, **components: str) -> Provenance:
    return Provenance(
        source="real_data",
        label=label,
        data_sources=data_sources or [],
        components=components,
        notes=["Derived from imported public dataset frames or labels."],
    )


def placeholder(label: str, **components: str) -> Provenance:
    return Provenance(
        source="placeholder",
        label=label,
        components=components,
        notes=["Uses placeholder data because the current sequence lacks the required real sensor/state stream."],
    )


def terrain_source(sequence_id: str, source_format: str | None = None) -> Provenance:
    if source_format == "rugd-style" or sequence_id.lower().startswith("rugd"):
        return real_data(
            "RUGD-style public semantic labels",
            data_sources=["RUGD-style RGB frames", "RUGD-style semantic masks"],
            model="tiny traversability classifier",
        )
    return synthetic("OR-WM demo semantic labels", model="tiny traversability classifier")


def trajectory_source(sequence_id: str) -> Provenance:
    if sequence_id.lower().startswith("tartan"):
        return real_data(
            "trajectory dataset with real poses/actions",
            data_sources=["TartanDrive-style pose/action stream"],
            model="TinyTrajGRU",
        )
    return placeholder(
        "placeholder pose stream",
        model="TinyTrajGRU",
        required_next="Import TartanDrive-style poses/actions before treating ADE/FDE as real.",
    )
