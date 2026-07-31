"""External API tool via RapidAPI: chemical element lookup.

RapidAPI is an API marketplace where a single key unlocks many different APIs.
This tool demonstrates that pattern by looking up a chemical element's properties
from a periodic-table API hosted on RapidAPI.

Setup (free):
  1. Create an account at rapidapi.com.
  2. Subscribe (free tier) to a periodic-table API, for example
     "Periodic Table of Elements" (host: periodic-table-of-elements1.p.rapidapi.com).
  3. Put your key in the RAPIDAPI_KEY environment variable.
  4. If you use a different periodic-table API, set RAPIDAPI_HOST (and optionally
     RAPIDAPI_URL) to match it.

Calls use the standard RapidAPI headers (X-RapidAPI-Key, X-RapidAPI-Host). The
tool fails soft: a missing key or a failed call returns a message, never a crash.
"""
import os
import requests
from langchain.tools import tool

# Default periodic-table API on RapidAPI. Override via env if you subscribe to a
# different one.
DEFAULT_HOST = "periodic-table-of-elements1.p.rapidapi.com"


@tool
def element_lookup(element: str) -> str:
    """Look up a chemical element's properties (atomic number, atomic mass, symbol,
    category, phase) from a periodic-table API via RapidAPI.
    'element' can be a name like 'Iron' or a symbol like 'Fe'."""
    key = os.getenv("RAPIDAPI_KEY")
    if not key:
        return "Element lookup unavailable: RAPIDAPI_KEY is not set. Subscribe to a periodic-table API on rapidapi.com and set the key."
    host = os.getenv("RAPIDAPI_HOST", DEFAULT_HOST)
    url = os.getenv("RAPIDAPI_URL", f"https://{host}/")
    try:
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
        # Different periodic-table APIs name the query differently, so send the
        # common ones; the API ignores the parameters it does not use.
        params = {"symbol": element, "name": element, "element": element}
        r = requests.get(url, headers=headers, params=params, timeout=12)
        if r.status_code != 200:
            return f"Element lookup unavailable (HTTP {r.status_code}). Check your RapidAPI subscription and RAPIDAPI_HOST."
        data = r.json()
        return _summarise(element, data)
    except Exception as e:
        return f"Element lookup failed: {e}"


def _summarise(element: str, data) -> str:
    """Pull the common element fields out of whatever shape the API returns."""
    # Some APIs return a list; take the first entry.
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return f"Element data for {element}: {str(data)[:500]}"

    def pick(*names):
        for n in names:
            for k, v in data.items():
                if k.lower().replace(" ", "").replace("_", "") == n:
                    return v
        return None

    name = pick("name", "atomname", "elementname") or element
    symbol = pick("symbol", "atomsymbol", "atomsymbole")
    number = pick("atomicnumber", "number")
    mass = pick("atomicmass", "atomicweight", "mass")
    category = pick("category", "phase", "atomphase", "group")
    parts = [f"{name}"]
    if symbol: parts.append(f"symbol {symbol}")
    if number is not None: parts.append(f"atomic number {number}")
    if mass is not None: parts.append(f"atomic mass {mass}")
    if category: parts.append(f"category/phase {category}")
    return "Element (via RapidAPI): " + ", ".join(str(p) for p in parts) + "."
