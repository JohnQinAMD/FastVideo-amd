#!/usr/bin/env python3
"""Extract PMC counters from rocprofv3 SQLite DB."""
import sqlite3, sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vsa_fwd_results.db"
db = sqlite3.connect(db_path)
cur = db.cursor()

# Find UUID from metadata table name
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_metadata_%'")
meta_table = cur.fetchone()[0]
u = meta_table.replace("rocpd_metadata_", "")

# PMC id -> name
cur.execute(f"SELECT id, name FROM rocpd_info_pmc_{u}")
pmc_id_to_name = {r[0]: r[1] for r in cur.fetchall()}

# kernel_symbol: id -> (kernel_name, sgpr, arch_vgpr, accum_vgpr)
cur.execute(f"SELECT id, kernel_name, sgpr_count, arch_vgpr_count, accum_vgpr_count FROM rocpd_info_kernel_symbol_{u}")
kern_info = {r[0]: (r[1], r[2], r[3], r[4]) for r in cur.fetchall()}

# dispatches
cur.execute(f"""SELECT id, kernel_id, event_id, start, end,
    grid_size_x, grid_size_y, grid_size_z, workgroup_size_x
    FROM rocpd_kernel_dispatch_{u} ORDER BY id""")
dispatches = cur.fetchall()

# PMC per event, aggregate across SEs
cur.execute(f"SELECT event_id, pmc_id, value FROM rocpd_pmc_event_{u}")
pmc_events = {}
for eid, pid, val in cur.fetchall():
    name = pmc_id_to_name.get(pid, f"pmc_{pid}")
    pmc_events.setdefault(eid, {})
    pmc_events[eid][name] = pmc_events[eid].get(name, 0) + val

# Print header
sep = "-" * 210
hdr = f"{'ID':>3} {'Kernel':<80} {'dur_us':>8} {'grid':>14} {'wg':>4} {'sgpr':>5} {'vgpr':>5} {'agpr':>5} {'MFMA_BF16':>12} {'VMEM':>12} {'WAIT_LDS':>12} {'WAIT_ANY':>12} {'W/M':>6}"
print(hdr)
print(sep)

for d in dispatches:
    did, kid, eid = d[0], d[1], d[2]
    dur_us = (d[4] - d[3]) / 1000.0
    grid = f"{d[5]}x{d[6]}x{d[7]}"
    wg = d[8]
    ki = kern_info.get(kid, ("?", 0, 0, 0))
    kname = ki[0][:78]
    pmcs = pmc_events.get(eid, {})
    mfma = pmcs.get("SQ_INSTS_VALU_MFMA_BF16", 0)
    vmem = pmcs.get("SQ_INSTS_VMEM", 0)
    wait_lds = pmcs.get("SQ_WAIT_INST_LDS", 0)
    wait_any = pmcs.get("SQ_WAIT_INST_ANY", 0)
    ratio = wait_any / mfma if mfma > 0 else -1
    print(f"{did:3d} {kname:<80} {dur_us:8.1f} {grid:>14} {wg:4d} {ki[1]:5d} {ki[2]:5d} {ki[3]:5d} {mfma:12.0f} {vmem:12.0f} {wait_lds:12.0f} {wait_any:12.0f} {ratio:6.1f}")

db.close()
