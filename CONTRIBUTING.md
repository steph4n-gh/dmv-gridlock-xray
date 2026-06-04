# Contributing Guidelines

Thank you for contributing to the **DMV Gridlock X-Ray** project. This document outlines the development standards, engineering requirements, and code submission guidelines that all contributors must follow.

---

## 🔒 Mandatory Development Standards & Rules

To ensure mathematical stability, system security, and data integrity, you **must** strictly adhere to the following rules:

### 1. The Post-Sprint Documentation Mandate (CRITICAL)
After each feature addition, bug fix, or optimization cycle, you **must** update and extend the mathematical architecture manual in [ARCHITECTURE.md](ARCHITECTURE.md).
*   Your descriptions must be technically rigorous and avoid high-level generalizations.
*   Every physical, topological, or congestion metric must be documented with its exact mathematical equations formatted in standard GFM Markdown (using standard LaTeX `$$` blocks for block equations and `$` for inline variables).
*   Do not leave TODOs, placeholders, or stub sections in documentation or comments.

### 2. Numerical Stability Floor
When modifying the mathematical friction matrix scaling, always enforce the physical clamp floor limit of `0.01` to prevent singularity collapses in the sparse eigensolver (`eigsh`):
$$f_i = \max(f_i, 0.01)$$
Do not allow friction values to drop below this stability threshold.

### 3. Symmetry Breaker
To prevent degenerate eigenvalue ties (multiplicity conflicts) in highly symmetric topologies, ensure that a microscopic deterministic noise factor scaled by $\tau$ ($2\pi$ or `math.tau`) remains injected into the friction array.

### 4. Zero-Downtime Daemon Rule
Do not kill, terminate, or restart the active background processes (`engine.py`, `server.py`) on live staging/production servers during audits or sprints unless explicitly instructed. They are active daemons powering the live stream.

### 5. Cross-Process Advisory Locking & Atomic Writes
Any file read/write operation involving shared state files (such as `sim_state.json` or `network_state.npz`) must adhere to strict concurrency guards:
*   **Advisory File Locks**: Use the `file_lock` context manager (using `fcntl.flock` with shared/exclusive locks and fallback try-except logic) to coordinate operations across the background engine, web server, and visualizer processes.
*   **Atomic Swapping**: Write file updates to a temporary staging file (e.g. `<filename>.tmp`), call `.flush()`, commit buffers to disk via `os.fsync(f.fileno())`, and swap using `os.replace` to prevent corrupted or half-written states.

---

## 🛠️ Verification & Compilation Guards

Before declaring a task or submission complete, you must run the following verification steps:

### 1. Syntax Compilation Guard
All modified source files must compile successfully under the virtual environment's Python compiler. Execute the following command from the project root:
```bash
./venv/bin/python -m py_compile engine.py server.py utils.py viz.py
```
This check must pass with an exit code of `0` and no syntax warnings.

### 2. Run the Test Suite
Ensure that all unit and integration tests pass successfully. Execute:
```bash
./venv/bin/python -m unittest discover -s tests
```
*Note: If `pytest` is configured in your development environment, you can run:*
```bash
./venv/bin/pytest
```
*All tests must be fully passing. If you write new code, you are expected to add corresponding tests in the `tests/` directory.*

---

## 📐 Style & Coding Guidelines

*   **Type Annotations**: All new functions and methods must include explicit PEP 484 type hints for all parameters and return values.
*   **Top-Level Imports**: Do not nest imports inside functions or loops. All import statements must be declared at the top of the file.
*   **Error Handling**: Avoid bare `except:` blocks. Always catch specific exceptions, log detailed stack traces to the appropriate log file (`server.log` or `engine.log`), and return sanitized, generic errors to the client.

---

## 🚀 Code Submission Protocol

1.  **Branching**: Create a descriptive feature branch from `main`:
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  **Verify**: Run the compilation guard and test suite to verify code correctness.
3.  **Document**: Update `ARCHITECTURE.md` with equations and physical phenomena mappings if new metrics or feeds were added.
4.  **Commit**: Write logical, concise commit messages.
5.  **Review**: Submit a Pull Request. Ensure that all automated checks (syntax compilation and unit tests) pass before requesting peer review.
