"""
Preprocessing script for MakeMore Argentina.
Cleans and prepares historico-nombres.csv and apellidos_cantidad_personas_provincia.csv
into makemore-compatible text files (one name per line, lowercase).
Also pre-computes aggregated data for the Streamlit app.
"""

import csv
import json
import re
import os
from collections import defaultdict

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
NOMBRES_CSV = os.path.join(ROOT_DIR, "historico-nombres.csv")
APELLIDOS_CSV = os.path.join(ROOT_DIR, "apellidos_cantidad_personas_provincia.csv")
OUTPUT_DIR = SCRIPT_DIR  # data/

# Thresholds
MIN_CANTIDAD_NOMBRES = 5     # ignore names with fewer than 5 people
MIN_CANTIDAD_APELLIDOS = 10  # already filtered in source (>10), but enforce


def clean_nombre(nombre: str) -> str | None:
    """
    Clean a single nombre entry. Returns None if should be discarded.
    Removes entries with '(presunto', normalizes to lowercase, strips whitespace.
    """
    if not nombre:
        return None
    # Discard entries with "(presunto" — these are data quality issues with DNI references
    if "(presunto" in nombre.lower() or "presunto" in nombre.lower():
        return None
    # Remove any remaining parenthetical content
    nombre = re.sub(r'\(.*?\)', '', nombre)
    # Strip and lowercase
    nombre = nombre.strip().lower()
    # Remove names that are just numbers or have digits
    if re.search(r'\d', nombre):
        return None
    # Remove names that are empty after cleaning
    if not nombre or len(nombre) < 2:
        return None
    # Remove names with weird characters (keep letters, spaces, accents, ñ, ü)
    # Valid chars: a-z, áéíóúñü, space, hyphen
    if not re.match(r'^[a-záéíóúñü\s\-]+$', nombre):
        return None
    # Collapse multiple spaces
    nombre = re.sub(r'\s+', ' ', nombre).strip()
    return nombre


def clean_apellido(apellido: str) -> str | None:
    """Clean a single apellido entry."""
    if not apellido:
        return None
    apellido = apellido.strip().lower()
    if len(apellido) < 2:
        return None
    if re.search(r'\d', apellido):
        return None
    # Valid chars for surnames
    if not re.match(r"^[a-záéíóúñü\s\-']+$", apellido):
        return None
    apellido = re.sub(r'\s+', ' ', apellido).strip()
    return apellido


def process_nombres():
    """
    Process historico-nombres.csv into:
    1. data/nombres_argentinos.txt  — unique names, one per line (makemore format)
    2. data/nombres_freq.txt        — names repeated by frequency (weighted, capped)
    3. data/nombres_stats.json      — aggregated stats for Streamlit
    """
    print("=" * 60)
    print("Processing historico-nombres.csv ...")
    print("=" * 60)

    # Aggregate: for each unique name, sum quantities across all years
    name_total = defaultdict(int)       # name -> total count across all years
    name_by_year = defaultdict(dict)    # name -> {year: count}
    discarded = 0
    total_rows = 0

    with open(NOMBRES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % 1_000_000 == 0:
                print(f"  ... processed {total_rows:,} rows")

            nombre_raw = row.get('nombre', '')
            cantidad = int(row.get('cantidad', 0))
            anio = int(row.get('anio', 0))

            nombre = clean_nombre(nombre_raw)
            if nombre is None:
                discarded += 1
                continue
            if cantidad < MIN_CANTIDAD_NOMBRES:
                discarded += 1
                continue

            name_total[nombre] += cantidad
            name_by_year[nombre][anio] = name_by_year[nombre].get(anio, 0) + cantidad

    print(f"\nTotal rows read: {total_rows:,}")
    print(f"Discarded: {discarded:,}")
    print(f"Unique clean names: {len(name_total):,}")

    # Sort by total frequency descending
    sorted_names = sorted(name_total.items(), key=lambda x: -x[1])

    # 1. Write unique names file (makemore format: one per line, lowercase)
    unique_path = os.path.join(OUTPUT_DIR, "nombres_argentinos.txt")
    with open(unique_path, 'w', encoding='utf-8') as f:
        for name, _ in sorted_names:
            f.write(name + '\n')
    print(f"Wrote {len(sorted_names):,} unique names to {unique_path}")

    # 2. Write frequency-weighted file (repeat names proportionally, capped to keep manageable)
    # Strategy: repeat each name ceil(log2(count)) times — balances frequency without exploding size
    import math
    freq_path = os.path.join(OUTPUT_DIR, "nombres_freq.txt")
    freq_count = 0
    with open(freq_path, 'w', encoding='utf-8') as f:
        for name, count in sorted_names:
            repeats = max(1, int(math.log2(count + 1)))
            for _ in range(repeats):
                f.write(name + '\n')
                freq_count += 1
    print(f"Wrote {freq_count:,} frequency-weighted entries to {freq_path}")

    # 3. Write stats JSON for Streamlit app
    top_100_all_time = [{"nombre": n, "cantidad": c} for n, c in sorted_names[:100]]

    # Top names per year (top 20 per year)
    top_by_year = {}
    year_totals = defaultdict(int)
    for name, years in name_by_year.items():
        for year, count in years.items():
            year_totals[year] += count

    # Build per-year rankings
    year_name_counts = defaultdict(list)
    for name, years in name_by_year.items():
        for year, count in years.items():
            year_name_counts[year].append((name, count))

    for year in sorted(year_name_counts.keys()):
        entries = sorted(year_name_counts[year], key=lambda x: -x[1])[:20]
        top_by_year[str(year)] = [{"nombre": n, "cantidad": c} for n, c in entries]

    stats = {
        "total_unique_names": len(name_total),
        "total_rows_processed": total_rows,
        "discarded_rows": discarded,
        "min_year": min(year_totals.keys()) if year_totals else 0,
        "max_year": max(year_totals.keys()) if year_totals else 0,
        "top_100_all_time": top_100_all_time,
        "top_by_year": top_by_year,
    }
    stats_path = os.path.join(OUTPUT_DIR, "nombres_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote stats to {stats_path}")

    # Print sample
    print("\nTop 20 names (all-time):")
    for name, count in sorted_names[:20]:
        print(f"  {name}: {count:,}")

    return sorted_names


def process_apellidos():
    """
    Process apellidos_cantidad_personas_provincia.csv into:
    1. data/apellidos_argentinos.txt  — unique surnames, one per line (makemore format)
    2. data/apellidos_freq.txt        — frequency-weighted file
    3. data/apellidos_stats.json      — aggregated stats for Streamlit
    """
    print("\n" + "=" * 60)
    print("Processing apellidos_cantidad_personas_provincia.csv ...")
    print("=" * 60)

    # Aggregate: for each apellido, sum across all provincias
    apellido_total = defaultdict(int)
    apellido_by_prov = defaultdict(dict)  # apellido -> {provincia: count}
    provincia_names = {}  # id -> name
    total_rows = 0
    discarded = 0

    with open(APELLIDOS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            apellido_raw = row.get('apellido', '')
            cantidad = int(row.get('cantidad', 0))
            prov_id = row.get('provincia_id', '')
            prov_nombre = row.get('provincia_nombre', '').strip()

            if prov_id and prov_nombre:
                provincia_names[prov_id] = prov_nombre

            apellido = clean_apellido(apellido_raw)
            if apellido is None:
                discarded += 1
                continue
            if cantidad < MIN_CANTIDAD_APELLIDOS:
                discarded += 1
                continue

            apellido_total[apellido] += cantidad
            apellido_by_prov[apellido][prov_nombre] = (
                apellido_by_prov[apellido].get(prov_nombre, 0) + cantidad
            )

    print(f"\nTotal rows read: {total_rows:,}")
    print(f"Discarded: {discarded:,}")
    print(f"Unique clean surnames: {len(apellido_total):,}")
    print(f"Provinces found: {len(provincia_names)}")
    for pid, pname in sorted(provincia_names.items(), key=lambda x: int(x[0])):
        print(f"  [{pid}] {pname}")

    sorted_apellidos = sorted(apellido_total.items(), key=lambda x: -x[1])

    # 1. Unique surnames file
    unique_path = os.path.join(OUTPUT_DIR, "apellidos_argentinos.txt")
    with open(unique_path, 'w', encoding='utf-8') as f:
        for apellido, _ in sorted_apellidos:
            f.write(apellido + '\n')
    print(f"Wrote {len(sorted_apellidos):,} unique surnames to {unique_path}")

    # 2. Frequency-weighted file
    import math
    freq_path = os.path.join(OUTPUT_DIR, "apellidos_freq.txt")
    freq_count = 0
    with open(freq_path, 'w', encoding='utf-8') as f:
        for apellido, count in sorted_apellidos:
            repeats = max(1, int(math.log2(count + 1)))
            for _ in range(repeats):
                f.write(apellido + '\n')
                freq_count += 1
    print(f"Wrote {freq_count:,} frequency-weighted entries to {freq_path}")

    # 3. Stats JSON
    top_100 = [{"apellido": a, "cantidad": c} for a, c in sorted_apellidos[:100]]

    top_by_prov = {}
    prov_apellido_counts = defaultdict(list)
    for apellido, provs in apellido_by_prov.items():
        for prov, count in provs.items():
            prov_apellido_counts[prov].append((apellido, count))

    for prov in sorted(prov_apellido_counts.keys()):
        entries = sorted(prov_apellido_counts[prov], key=lambda x: -x[1])[:20]
        top_by_prov[prov] = [{"apellido": a, "cantidad": c} for a, c in entries]

    stats = {
        "total_unique_surnames": len(apellido_total),
        "total_rows_processed": total_rows,
        "discarded_rows": discarded,
        "provinces": {pid: pname for pid, pname in sorted(provincia_names.items(), key=lambda x: int(x[0]))},
        "top_100_national": top_100,
        "top_by_province": top_by_prov,
    }
    stats_path = os.path.join(OUTPUT_DIR, "apellidos_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote stats to {stats_path}")

    print("\nTop 20 surnames (national):")
    for apellido, count in sorted_apellidos[:20]:
        print(f"  {apellido}: {count:,}")

    return sorted_apellidos


if __name__ == '__main__':
    process_nombres()
    process_apellidos()
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
