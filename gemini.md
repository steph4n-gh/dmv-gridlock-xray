# Gemini Agent Instructions & Standards: DMV Gridlock X-Ray

Welcome, Antigravity/Gemini developer. This file defines the core standards, engineering practices, and documentation rules for the **DMV Gridlock X-Ray** project. You are pair programming with the USER and must adhere strictly to these principles at all times.

---

## 1. The Post-Sprint Documentation Mandate (CRITICAL)
After **each and every sprint**, feature addition, or optimization cycle, you **MUST** update and extend the codebase documentation in [ARCHITECTURE.md](file:///Volumes/Storage/wmataworkspace/bustracker/ARCHITECTURE.md) to match the standard set in the May 2026 mathematical audit.

Your documentation must be:
*   **Exceptional & Highly Comprehensive**: Avoid high-level hand-waving or generic descriptions. Write with extreme technical rigor.
*   **Truthful & "No Bullshit"**: Document exactly what is implemented in the codebase. Never exaggerate capabilities or invent processes that do not exist.
*   **Zero Placeholders**: Never leave TODOs, placeholders, or stub sections in documentation, comments, or files.
*   **Mathematically Rigorous**: Every physical and topological metric must be documented with its exact mathematical equations formatted in standard GFM Markdown (using standard LaTeX `$$` blocks for block equations and `$` for inline variables).

---

## 2. Ingested Input Registry ("For & Why" Standard)
Every data feed ingested by `engine.py` (e.g., GTFS-RT, GBFS, REST GeoJSON, weather forecasts) must be meticulously logged and explained in [ARCHITECTURE.md](file:///Volumes/Storage/wmataworkspace/bustracker/ARCHITECTURE.md) under the following schema:
1.  **Protocol & Frequency**: Document the data transfer format (e.g., Protocol Buffers `.pb`, JSON REST, Web Mercator GeoJSON) and polling cycle duration.
2.  **Key Fields Parsed**: List the exact variables, keys, and objects extracted from the raw payload.
3.  **Physical Phenomenon**: Define what real-world state or urban event this represents.
4.  **Purpose & "Why"**: Detail the exact mathematical or structural reason the feed is ingested and how it feeds into the friction or centrality calculation.

---

## 3. Telemetry & GUI Output Registry ("What & Why" Standard)
Every metric, visualization, panel, or widget displayed on the dashboard interface (`index.html`) or logged by the engine must be meticulously logged and explained in [ARCHITECTURE.md](file:///Volumes/Storage/wmataworkspace/bustracker/ARCHITECTURE.md) under the following schema:
1.  **What it Displays**: The visual formatting, units of measurement, or ranges shown.
2.  **Why it is Displayed**: The mathematical calculation, underlying equations, active variables, and physical scaling factors.
3.  **Operational Utility**: Explain exactly how a metropolitan transit dispatcher or municipal planner utilizes this telemetry to make real-time operational decisions.

---

## 4. Codebase Safety & Daemon Rules
*   **Zero-Downtime Daemon Rule**: Do not kill, terminate, or restart the active background processes (`engine.py`, `server.py`) during audits or sprints unless explicitly instructed. They are active daemons powering the live stream.
*   **Syntax Compilation Guard**: Before declaring a sprint complete, always verify syntactic correctness using the virtual environment's compiler:
    ```bash
    ./venv/bin/python -m py_compile engine.py
    ```
*   **Numerical Stability Floor**: When editing the mathematical friction matrix scaling, always enforce the physical clamp floor (`0.01` floor limit) to prevent singularity collapses in the sparse eigensolver (`eigsh`).
*   **Symmetry Breaker**: Ensure the `math.tau` microscopic deterministic noise remains injected into the friction array to prevent degenerate eigenvalue ties in symmetric topologies.
