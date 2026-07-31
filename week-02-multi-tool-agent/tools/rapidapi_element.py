"""External API tool: chemical element lookup, via RapidAPI (with a free fallback).

RapidAPI is an API marketplace where a single key unlocks many APIs. This tool
demonstrates that pattern: if a RAPIDAPI_KEY is set, it calls a periodic-table API
hosted on RapidAPI (standard X-RapidAPI-Key / X-RapidAPI-Host headers). If no key
is set, it falls back to a free public periodic-table API, so the tool still works
out of the box. Either way it fails soft.

RapidAPI setup (free): create an account at rapidapi.com, subscribe to a
periodic-table API (default host: periodic-table-of-elements1.p.rapidapi.com),
put the key in RAPIDAPI_KEY, and set RAPIDAPI_HOST if you use a different one.
"""
import os
import requests
from langchain.tools import tool

DEFAULT_HOST = "periodic-table-of-elements1.p.rapidapi.com"
FREE_API = "https://api.periodictableofelements.org/elements/"
_UA = {"User-Agent": "materials-research-assistant"}


@tool
def element_lookup(element: str) -> str:
    """Look up a chemical element's properties (atomic number, atomic mass, symbol,
    category, state at room temperature) by name (for example 'Iron') or symbol
    (for example 'Fe')."""
    key = os.getenv("RAPIDAPI_KEY")
    if key:
        result = _via_rapidapi(element, key)
        if result:
            return result
        # RapidAPI failed; fall through to the free API rather than error out.
    return _via_free_api(element)


def _via_rapidapi(element: str, key: str):
    """Primary path: a periodic-table API on RapidAPI. Returns a string, or None on failure."""
    host = os.getenv("RAPIDAPI_HOST", DEFAULT_HOST)
    url = os.getenv("RAPIDAPI_URL", f"https://{host}/")
    try:
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
        # Different periodic-table APIs name the query differently; send the common
        # ones and let the API use what it recognises.
        params = {"symbol": element, "name": element, "element": element}
        r = requests.get(url, headers=headers, params=params, timeout=12)
        if r.status_code != 200:
            return None
        return "Element (via RapidAPI): " + _summarise_generic(element, r.json())
    except Exception:
        return None


def _via_free_api(element: str) -> str:
    """Fallback path: a free public periodic-table API (no key needed)."""
    try:
        r = requests.get(FREE_API, headers=_UA, timeout=12)
        if r.status_code != 200:
            return f"Element lookup unavailable (HTTP {r.status_code})."
        q = element.strip().lower()
        match = next(
            (e for e in r.json()
             if str(e.get("symbol", "")).lower() == q or str(e.get("name", "")).lower() == q),
            None,
        )
        if not match:
            return f"No element found for '{element}'. Use a name like 'Iron' or a symbol like 'Fe'."
        return (
            f"{match.get('name')} ({match.get('symbol')}): atomic number {match.get('atomic_number')}, "
            f"atomic mass {match.get('atomic_mass')}, category {match.get('category')}, "
            f"state at room temperature {match.get('state_at_room_temp')}."
        )
    except Exception as e:
        return f"Element lookup failed: {e}"


def _summarise_generic(element: str, data) -> str:
    """Pull common element fields out of whatever shape a RapidAPI response uses."""
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return str(data)[:400]

    def pick(*names):
        for n in names:
            for k, v in data.items():
                if k.lower().replace(" ", "").replace("_", "") == n:
                    return v
        return None

    name = pick("name", "atomname", "elementname") or element
    parts = [str(name)]
    for label, keys in [
        ("symbol", ("symbol", "atomsymbol", "atomsymbole")),
        ("atomic number", ("atomicnumber", "number")),
        ("atomic mass", ("atomicmass", "atomicweight", "mass")),
        ("category/phase", ("category", "phase", "atomphase")),
    ]:
        val = pick(*keys)
        if val is not None:
            parts.append(f"{label} {val}")
    return ", ".join(parts) + "."
