"""Readable display labels; stored evidence remains exact."""
from decimal import Decimal, ROUND_HALF_UP


def format_progress(progress, unit=None):
    if progress is None:
        return "unknown progress"
    value = Decimal(str(progress))
    suffix = ""
    for threshold, candidate in ((10**9, "B"), (10**6, "M"), (10**3, "K")):
        if abs(value) >= threshold:
            value /= threshold
            suffix = candidate
            break
    number = format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f").rstrip("0").rstrip(".") if value % 1 else str(int(value))
    unit = {"environment_steps": "steps", "optimizer_updates": "updates"}.get(unit, unit or "units")
    return f"{number}{suffix} {unit}"
