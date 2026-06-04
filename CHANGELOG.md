# Changelog

All notable changes to the **DMV Gridlock X-Ray** project are documented in this file. This project adheres to a structured, milestone-based release cycle focusing on mathematical modeling stability, multi-operator data fusion, concurrency hardening, and strict API security constraints.

---

## Milestone 3: Engine Refactoring & Cleanups — 2026-06-01

This milestone focused on refactoring the core simulation processing loop to improve maintainability, introducing robust cross-process synchronization for shared state files, and hardening mathematical solvers against topological edge cases.

### Added
*   **Advisory File Locking**: Integrated `fcntl.flock` cross-process locking (`sim_state.lock` with shared and exclusive modes) for all reads and writes to `sim_state.json` in `server.py`, `engine.py`, and `viz.py`. Includes a graceful try-except fallback for non-POSIX (Windows) environments.
*   **Dense Eigensolver Fallback**: Added an automatic fallback to a dense solver (`scipy.linalg.eigh(A.toarray())`) inside the spectral bisection routine when the giant connected component size $N \le 3$. This prevents sparse solver crashes (`scipy.sparse.linalg.eigsh` type errors) in tiny or isolated graph environments.

### Changed
*   **Ingestion Loop Decomposition**: Modularized the monolithic `main_loop` in `engine.py` into five decoupled, focused helper functions:
    1.  `compute_friction_field()`
    2.  `detect_fractures()`
    3.  `adjudicate_predictions()`
    4.  `export_visualization_data()`
    5.  `print_console_dashboard()`
*   **Non-Blocking Console Clears**: Replaced slow and synchronous subshell calls (`os.system('clear')`) in the console HUD rendering with the non-blocking ANSI escape sequence `\033[2J\033[H` to reduce processor strain and avoid terminal flickering.
*   **Atomic Write Swapping**: Hardened all `sim_state.json` mutations to use an atomic write-and-rename pattern (`tempfile` staging, flushing, flushing OS buffers via `os.fsync`, and replacing via `os.replace`) to prevent partial or corrupted reads by concurrent web clients.

### Fixed
*   **Exception Safety**: Replaced all bare `except:` catch-all blocks in `engine.py` with explicit, typed exception handlers that log details and traceback snippets to prevent silent failure states.

---

## Milestone 2: Security & Concurrency Hardening — 2026-05-20

This milestone introduced multi-threaded HTTP server handling, strict concurrency guards, constant-time authentication, route parameter sanitization, and path traversal blocks.

### Added
*   **Multi-Threaded HTTP Server**: Upgraded the web server in `server.py` to inherit from `socketserver.ThreadingTCPServer`, enabling parallel execution of path routing operations and request handlers.
*   **Concurrency Locks**: Integrated global reentrant locks (`_cache_lock`, `_sim_lock`, `_rate_limit_lock`) to synchronize access to route cache objects, simulation inject lists, and rate limit histories.
*   **Double-Checked Locking**: Implemented the double-checked locking design pattern inside `load_and_augment_graph()` in `server.py` to prevent duplicate loading/compiling of the graph data structure across parallel requests.
*   **Dijkstra Engine Unification**: Consolidating routing logic by extracting Dijkstra calculations, route segment assembly, transit mode breakdowns, and transfer penalty processing into a single private `_compute_route` helper, now shared by `/api/route` and `/api/bridge`.
*   **Constant-Time Authorization**: Added `is_authorized()` to administrative mutation endpoints (`/api/inject` and `/api/clear`), validating Bearer tokens in constant-time using `hmac.compare_digest` to prevent timing attacks.
*   **Path Traversal Prevention**: Overrode `do_GET` to normalize request paths via `posixpath.normpath` and enforce strict directory boundary checks. Only files listed in a defined static asset whitelist are allowed to be served.
*   **Favicon Fallback Handler**: Added automatic serving of a default/dummy favicon to satisfy browser queries and prevent handler exceptions.
*   **Generic Error Sanitization**: Integrated a centralized `handle_error` wrapper that intercepts internal exceptions, logs full stack traces to `server.log`, and returns clean, sanitized JSON error objects without revealing database locations or code structure.
*   **Security Test Coverage**: Expanded the test suite with 9 specialized unit and integration tests verifying authorization headers, rate limits (HTTP 429), directory traversal blocks, and error sanitizers.

---

## Milestone 1: Consolidated Utilities & Coordinate Foundation — 2026-05-05

This milestone established a unified geometric and mathematical helper foundation, corrected coordinate validation constraints, and eliminated duplicate structures.

### Added
*   **Consolidated Utilities Module**: Created `utils.py` containing Potomac and Anacostia River geometries, coordinate math for intersection detection (`ccw`, `segments_intersect`, `crosses_river`), `smooth_floor` softplus decay function, and `validate_coordinates`.
*   **DMV Bounding Box Checker**: Introduced `validate_coordinates()` to strictly check bounds ($38.0 \le \text{lat} \le 40.0$ and $-78.0 \le \text{lon} \le -76.0$), filtering out NaN, infinity, and non-numeric formats.

### Changed
*   **De-duplication**: Removed duplicate copies of river boundary segments and intersection math from `engine.py` and `server.py`, routing all math checks through the centralized `utils.py` module.
*   **Namespace Cleanups**: Relocated nested internal imports (e.g. `import datetime`, `import traceback`, `import json`) to the top level of the namespace in both `engine.py` and `server.py`.

### Fixed
*   **Falsy Coordinate Bug**: Corrected the coordinate check in `/api/route` that evaluated `0.0` as falsy (and falsely returned a `"Missing coordinates"` error payload), allowing valid but out-of-DMV coordinates to proceed to the bounding box checker where they are safely validated and rejected as out-of-bounds.
*   **Test Suite Integration**: Modified `tests/test_suite.py` to import and verify `utils.py` functions and added server endpoint validators.
