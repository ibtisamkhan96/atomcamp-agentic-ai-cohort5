"""Custom Python tool: convert common materials units.

This is the assignment's required 'custom Python function' tool. It is pure
Python with no external API. It also fits the wider theme of harmonising messy
materials data, where mixed units are one of the most common problems.
"""
from langchain.tools import tool

# Base units: pressure/stress in pascals, density in kg/m3.
_PRESSURE = {
    "pa": 1.0, "kpa": 1e3, "mpa": 1e6, "gpa": 1e9,
    "bar": 1e5, "atm": 101325.0, "psi": 6894.757293, "ksi": 6.894757293e6,
}
_DENSITY = {
    "kg/m3": 1.0, "kg/m^3": 1.0, "g/cm3": 1000.0, "g/cm^3": 1000.0,
    "g/cc": 1000.0, "lb/in3": 27679.9047, "lb/in^3": 27679.9047,
}
_TEMP_NAMES = {"c", "celsius", "degc", "k", "kelvin", "f", "fahrenheit", "degf"}


def _norm(u: str) -> str:
    """Normalise a unit string: lowercase, strip spaces and the degree symbol."""
    return u.strip().lower().replace("°", "").replace(" ", "")


def _temp_to_c(v: float, u: str):
    if u in ("c", "celsius", "degc"):
        return v
    if u in ("k", "kelvin"):
        return v - 273.15
    if u in ("f", "fahrenheit", "degf"):
        return (v - 32.0) * 5.0 / 9.0
    return None


def _temp_from_c(c: float, u: str):
    if u in ("c", "celsius", "degc"):
        return c
    if u in ("k", "kelvin"):
        return c + 273.15
    if u in ("f", "fahrenheit", "degf"):
        return c * 9.0 / 5.0 + 32.0
    return None


def _convert(value: float, from_unit: str, to_unit: str) -> str:
    """The pure conversion logic, kept separate so it is easy to test."""
    f, t = _norm(from_unit), _norm(to_unit)

    # Temperature is non-linear, so handle it on its own.
    if f in _TEMP_NAMES or t in _TEMP_NAMES:
        if f not in _TEMP_NAMES or t not in _TEMP_NAMES:
            return f"Cannot convert between '{from_unit}' and '{to_unit}': different categories."
        out = _temp_from_c(_temp_to_c(value, f), t)
        return f"{value} {from_unit} = {round(out, 4)} {to_unit}"

    # Pressure / stress.
    if f in _PRESSURE and t in _PRESSURE:
        out = value * _PRESSURE[f] / _PRESSURE[t]
        return f"{value} {from_unit} = {round(out, 6)} {to_unit}"

    # Density.
    if f in _DENSITY and t in _DENSITY:
        out = value * _DENSITY[f] / _DENSITY[t]
        return f"{value} {from_unit} = {round(out, 6)} {to_unit}"

    return (f"Cannot convert '{from_unit}' to '{to_unit}'. Supported units: "
            "pressure or stress (Pa, kPa, MPa, GPa, bar, atm, psi, ksi), "
            "density (kg/m3, g/cm3, lb/in3), temperature (C, K, F).")


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a materials value between units.
    Supports pressure or stress (Pa, kPa, MPa, GPa, bar, atm, psi, ksi),
    density (kg/m3, g/cm3, lb/in3), and temperature (C, K, F).
    Provide the numeric value and the two unit strings."""
    try:
        return _convert(float(value), from_unit, to_unit)
    except Exception as e:
        return f"Conversion failed: {e}"
