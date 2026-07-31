"""External API tool #1: the Materials Project.

Looks up computed properties of inorganic materials by Materials Project ID
(for example mp-149) or by chemical formula (for example SiO2). Needs a free
MP_API_KEY from https://materialsproject.org/api. Fails soft if the key is
missing or the call errors, so it never crashes the agent loop.
"""
import os
import re
from langchain.tools import tool

_FIELDS = [
    "material_id", "formula_pretty", "band_gap", "density",
    "formation_energy_per_atom", "is_stable", "symmetry",
]


@tool
def materials_project_lookup(query: str) -> str:
    """Look up computed properties of an inorganic material from the Materials Project.
    Accepts a Materials Project ID like 'mp-149' or a chemical formula like 'SiO2'.
    Returns band gap, density, formation energy, stability and crystal system."""
    key = os.getenv("MP_API_KEY")
    if not key:
        return "Materials Project lookup unavailable: MP_API_KEY is not set."
    try:
        from mp_api.client import MPRester
        q = query.strip()
        with MPRester(key) as mpr:
            if re.fullmatch(r"mp-\d+", q):
                docs = mpr.materials.summary.search(material_ids=[q], fields=_FIELDS)
            else:
                docs = mpr.materials.summary.search(formula=q, fields=_FIELDS)
        if not docs:
            return f"No Materials Project entry found for '{query}'."
        d = docs[0]
        crystal = getattr(getattr(d, "symmetry", None), "crystal_system", None)
        return (
            f"Materials Project {d.material_id} ({d.formula_pretty}): "
            f"band gap {d.band_gap} eV, density {round(float(d.density), 3)} g/cm3, "
            f"formation energy {round(float(d.formation_energy_per_atom), 4)} eV/atom, "
            f"stable: {d.is_stable}, crystal system: {crystal}. "
            "Note: values are computed with DFT, so treat band gaps as lower bounds."
        )
    except Exception as e:
        return f"Materials Project lookup failed: {e}"
