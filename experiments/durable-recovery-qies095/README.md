# QIES094-095 model-free durable recovery experiments

Experimental preservation milestone, not production recovery code. Base: 8fac93bb2bfc4b2d7069c4044f2f71cd25f80388. No engine file is changed by this milestone. QIES086 is not included.

The exact original QIES094 storage prototype is under qies095-preserved/qies094-prototype/. The exact QIES095 coordinator, tests and synthetic reference vector are under qies095-prototype/. Original reports and the integration contract are under reports/. Source inventories and original worktree identities are under provenance/; absolute paths there describe the original laptop and are provenance, not portable launch instructions.

Fresh verification passed 23 test groups and 557 fault outcomes (309 original plus 248 replay outcomes). Both original SHA-256 inventories and all 701 QIES094 original files were verified locally before packaging. The complete historical synthetic run directories/archives and temporary products are excluded; source-sha256.json is the historical inventory, not a claim that every inventoried run file is packaged. SHA256SUMS verifies exactly this package's included files.

Run from this directory:

    sha256sum -c SHA256SUMS
    ./pandacode-run.sh

Python standard library only. The wrapper differs from the original only to create the initially absent output directory; the exact original wrapper remains in provenance/. Each execution creates a new ignored output directory. Tests use small synthetic files and terminate/reap only their own harmless forked children. They do not load model weights, run inference or alter system settings/guards.

Durable decisions are separate from evaluated state and external delivery. No exactly-once stdout guarantee. Native 48-layer state, stop/UTF-8 protocol completion, journal repair, retention and real-engine/oracle qualification remain blocked. Read the reports for the precise limits. No production integration or merge is authorized by this archive.
