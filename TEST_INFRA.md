# DMV Gridlock X-Ray Montgomery County RideOn - Test Infrastructure (TEST_INFRA.md)

This document outlines the testing philosophy, feature inventory, test architecture, and coverage thresholds for the Montgomery County RideOn integration.

---

## 1. Test Philosophy
The DMV Gridlock X-Ray system is a high-availability, real-time mathematical topology and pathfinding daemon. As such, the test infrastructure must ensure that the core calculations (algebraic connectivity, Dijkstra routing) remain numerically stable and correct under both real-time dynamic ingest conditions and offline sandbox environments.

Key tenets:
- **Requirement-Driven**: Every test maps back to a specific requirement (R1-R5) defined in the integration scope.
- **Deterministic & Offline**: 100% of the E2E test suite runs offline without making external HTTP queries, ensuring high reproducibility and preventing external endpoint dependency flaky failures.
- **Strict Rate Limit Modeling**: Rate limiting is tested using virtual clocks to ensure exact boundary behaviors (down to millisecond levels) are checked.
- **Topology Isolation**: Tests isolate their state by backing up, mocking, and restoring runtime files (`network_state.npz`, `stops_list.json`, `metro_tunnels.json`) during the execution lifecycle.

---

## 2. Feature Inventory & Requirements Mapping

### R1. Dynamic RideOn GTFS-RT Ingestion
- **R1.1**: Ingestion and parsing of RideOn vehicle position messages.
- **R1.2**: Ingestion and parsing of RideOn trip updates containing departure/arrival delays.
- **R1.3**: Proper ID prefixing (`rideon_`) for vehicles, routes, and trips.
- **R1.4**: Integration of RideOn speeds and delays into global engine state.
- **R1.5**: Graceful fallback and clamping for extreme speed inputs or missing fields.

### R2. Strict API Rate Limit Enforcement
- **R2.1**: Maximum 3 requests per minute per unique link URL.
- **R2.2**: Minimum 20 seconds interval between consecutive requests to the same URL.
- **R2.3**: Maximum 10 requests per minute overall across all URLs.
- **R2.4**: JSON-based state file persistence and parsing.
- **R2.5**: Corrupted state file detection and recovery.

### R3. Static Topology Fusion
- **R3.1**: Merging and prefixing RideOn static stops and shapes with WMATA data.
- **R3.2**: Giant Connected Component (GCC) topological subgraph extraction.
- **R3.3**: KDTree stop coordinate snapping and geodesic distance calculations.
- **R3.4**: Enforcement of walking transfer distances limits ($\le 300$ meters).

### R4. UI Visual Extensions & Dead-Reckoning
- **R4.1**: Separate Plotly trace config mappings (Trace 9 for WMATA, Trace 22 for RideOn).
- **R4.2**: Client-side dead-reckoning movement calculations utilizing bearing and speed.
- **R4.3**: Badge and operator label rendering extraction based on prefixed stop/vehicle IDs.
- **R4.4**: Pre-renderer table regex integer sorting for alphanumeric stops.

### R5. Intermodal Router Enhancements
- **R5.1**: Dijkstra calculations over the fused, dual-operator sparse Graph Laplacian.
- **R5.2**: Weather-penalized virtual transfer edges at intermodal snap nodes.
- **R5.3**: Route direction text instruction generation indicating provider transitions.
- **R5.4**: Dijkstra leg grouping by operator.

---

## 3. Test Architecture

The E2E Test Suite is structured around a **4-Tier Test Design**:

```
+-------------------------------------------------------------+
| TIER 4: Real-World Scenarios (e.g. Intermodal Routing)      |
+-------------------------------------------------------------+
| TIER 3: Cross-Feature Combinations (e.g. Storm Routing)     |
+-------------------------------------------------------------+
| TIER 2: Boundary & Corner Cases (e.g. Clamping, Dropout)    |
+-------------------------------------------------------------+
| TIER 1: Feature Coverage & Happy Paths (R1-R5 unit tests)   |
+-------------------------------------------------------------+
```

### Tiers Breakdown
- **Tier 1: Feature Coverage**: At least 5 tests per feature covering happy paths (25 tests total).
- **Tier 2: Boundary & Corner Cases**: Limits, empty/invalid inputs, dropouts, rate limit checks (32 tests total).
- **Tier 3: Cross-Feature Combinations**: Pairwise coverage of major feature interactions (5 tests total).
- **Tier 4: Real-World Application Scenarios**: Multimodal transit use cases, routing shifts, and offline resilience (5 tests total).

---

## 4. Coverage Thresholds
- **Total Test Cases**: 67 E2E and unit tests.
- **Pass Rate**: 100% required.
- **Execution Speed**: $< 1.0$ seconds for all 67 tests to encourage pre-commit runs.
- **External Network Dependence**: 0 HTTP requests.

---

## 5. Execution Guide
To run the E2E test suite in the virtual environment:
```bash
./venv/bin/python -m unittest tests/test_suite.py
```
