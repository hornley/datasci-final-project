import csv
import random
from collections import defaultdict

INPUT = "cs_students.csv"
OUTPUT = "cs_students.csv"
SEED = 42

rng = random.Random(SEED)

with open(INPUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("Dataset is empty")

# Build project pools by domain.
domain_to_projects = defaultdict(set)
all_projects = set()
for row in rows:
    domain = row["Interested Domain"].strip()
    project = row["Projects"].strip()
    if project:
        domain_to_projects[domain].add(project)
        all_projects.add(project)

all_projects = sorted(all_projects)

for row in rows:
    domain = row["Interested Domain"].strip()
    base_project = row["Projects"].strip()

    multi_projects = []
    if base_project:
        multi_projects.append(base_project)

    # Add one additional matching project when available.
    matching_pool = [p for p in sorted(domain_to_projects[domain]) if p != base_project]
    if matching_pool:
        multi_projects.append(rng.choice(matching_pool))

    # Add one non-matching project to inject noise and realism.
    non_matching_pool = [p for p in all_projects if p not in domain_to_projects[domain]]
    if non_matching_pool:
        multi_projects.append(rng.choice(non_matching_pool))

    # Guarantee more than one project if pools are limited.
    if len(multi_projects) < 2:
        fallback_pool = [p for p in all_projects if p != base_project]
        if fallback_pool:
            multi_projects.append(rng.choice(fallback_pool))

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for p in multi_projects:
        if p and p not in seen:
            deduped.append(p)
            seen.add(p)

    row["Projects"] = "; ".join(deduped)

fieldnames = list(rows[0].keys())
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {len(rows)} rows with multi-project values in {OUTPUT}")
