"""External API tool: OPTIMADE multi-database structure search.

OPTIMADE is a shared standard that many materials databases speak (Materials
Project, OQMD, COD, Alexandria and more). This single tool queries all of them
at once with one filter, so the agent can discover materials broadly and then
call materials_project_lookup for the deep properties of any Materials Project
hit (the OPTIMADE results include mp- ids).

Providers go up and down, so each is queried with a short timeout and skipped on
failure. This keeps the agent responsive even when a database is offline.
"""
import requests
from langchain.tools import tool

# Database name -> its OPTIMADE /structures endpoint.
_PROVIDERS = {
    "Materials Project": "https://optimade.materialsproject.org/v1/structures",
    "OQMD": "https://oqmd.org/optimade/v1/structures",
    "COD": "https://www.crystallography.net/cod/optimade/v1/structures",
    "Alexandria": "https://alexandria.icams.rub.de/pbe/v1/structures",
}
_HEADERS = {"User-Agent": "Mozilla/5.0 (materials-research-assistant)"}


def _build_filter(elements: str, nelements: int) -> str:
    """Turn 'Ti,O' into an OPTIMADE filter like: elements HAS ALL "Ti","O"."""
    symbols = [e.strip() for e in elements.split(",") if e.strip()]
    if not symbols:
        return ""
    quoted = ",".join(f'"{s}"' for s in symbols)
    flt = f"elements HAS ALL {quoted}"
    if nelements and nelements > 0:
        flt += f" AND nelements={nelements}"
    return flt


def _query_provider(name: str, url: str, flt: str, max_results: int) -> str:
    """Query one OPTIMADE provider. Fails soft: returns a message, never raises."""
    try:
        params = {"filter": flt, "page_limit": max_results}
        r = requests.get(url, params=params, headers=_HEADERS, timeout=12)
        if r.status_code != 200:
            return f"{name}: unavailable (HTTP {r.status_code})."
        data = r.json()
        rows = data.get("data", []) or []
        if not rows:
            return f"{name}: no matches."
        total = data.get("meta", {}).get("data_returned")
        samples = []
        for item in rows[:max_results]:
            attrs = item.get("attributes", {})
            formula = (attrs.get("chemical_formula_reduced")
                       or attrs.get("chemical_formula_descriptive") or "?")
            samples.append(f"{formula} ({item.get('id')})")
        total_txt = f"{total} total matches" if total is not None else f"{len(rows)} matches"
        return f"{name}: {total_txt}. Examples: " + ", ".join(samples) + "."
    except Exception as e:
        return f"{name}: query failed ({type(e).__name__})."


@tool
def optimade_search(elements: str, nelements: int = 0, max_results: int = 4) -> str:
    """Search many materials databases at once (Materials Project, OQMD, COD, Alexandria)
    for materials containing given elements, using the OPTIMADE standard.
    'elements' is a comma-separated list of chemical symbols, for example 'Ti,O'.
    Set 'nelements' to restrict the count of distinct elements (for example 2 for binaries).
    Returns matches per database. For any Materials Project result, pass its id (like mp-149)
    to materials_project_lookup to get detailed properties."""
    flt = _build_filter(elements, nelements)
    if not flt:
        return "Please provide at least one element symbol, for example 'Ti,O'."
    results = [_query_provider(name, url, flt, max_results) for name, url in _PROVIDERS.items()]
    return "\n".join(results)
