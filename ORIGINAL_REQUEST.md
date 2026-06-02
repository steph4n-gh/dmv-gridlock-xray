# Original User Request

## Initial Request — 2026-06-02T18:30:16Z

Integrate the Montgomery County RideOn bus network into the DMV Gridlock X-Ray system. This extends the live mathematical topology to ingest real-time RideOn feeds, merge static topologies, enforce strict client-side API rate limits, and display RideOn buses and routes on the 3D dashboard with intermodal routing support.

Working directory: /Volumes/Storage/wmataworkspace/bustracker
Integrity mode: development

Credentials for building and testing (configure via environment variables, do not hardcode in repository files):
- RIDEON_API_KEY: D35F41AC51604C69AD238C92FB4CBC2123033DDC398646F7A02F368F
- RIDEON_CLIENT_ID: 94
- WMATA_API_KEY: zx6enf74ehb638yn822yapbc

## Requirements

### R1. Dynamic RideOn GTFS-RT Ingestion
- Ingest real-time RideOn vehicle positions and trip updates using the Protocol Buffer feed:
  `http://rideon.app/protobuf/GetGtfsRealtime?apiKey={RIDEON_API_KEY}&ClientId={RIDEON_CLIENT_ID}`
- Parse vehicle speeds, bearings, trip delays, and locations, adapting the engine's concurrency-scaled density penalty and Ground Zero alerts to include RideOn vehicles.

### R2. Strict API Rate Limit Enforcement
- Implement a rigid, state-persisted rate limiting governor inside `engine.py`.
- Limit API calls to **no more than 3 requests per minute per link** (minimum polling interval of 20 seconds).
- Ensure total calls per API key do not exceed **10 requests per minute** overall.
- Handle potential connection dropouts gracefully without crashing the processing loop.

### R3. Static Topology Fusion
- Download the public Montgomery County RideOn static GTFS data feed.
- Merge the RideOn stops and route shapes with the existing WMATA topology to create a single, unified sparse Graph Laplacian matrix.
- Ensure stop snapping (KDTree) maps RideOn and WMATA stop nodes correctly.

### R4. UI Visual Extensions & Dead-Reckoning
- Update `index.html` and the buses/trains dashboards to visualize RideOn routes and vehicles.
- Enable smooth client-side dead-reckoning animation for RideOn buses.
- Style RideOn elements clearly (e.g., using a distinct RideOn color scheme or line identifiers).

### R5. Intermodal Router Enhancements
- Expand `/api/route` and `/api/bridge` to support paths transitioning between WMATA and RideOn segments.
- Apply virtual transfer edges and transfer penalties ($300.0$ seconds baseline, scaled by weather) at transition nodes where a passenger walks between WMATA and RideOn stops.

## Acceptance Criteria

### API Rate Limiting Guardrails
- [ ] No request rate limit errors (HTTP 429 or permission errors) occur during a 10-minute active run.
- [ ] Log telemetry verifying that the time between consecutive RideOn polls is strictly $\ge 20$ seconds.

### Data Ingestion and Topological Mesh
- [ ] RideOn stop nodes are successfully incorporated into the Graph Laplacian.
- [ ] Live RideOn buses appear and animate on the WebGL map.

### Intermodal Dijkstra Router
- [ ] Paths between a WMATA stop and a RideOn stop are solved and return directions indicating the physical walk transfer point.

## Follow-up — 2026-06-02T18:41:04Z

The user has requested that we ensure all technical documentation is updated and streamlined, specifically involving technical writers. 

Please proceed with completing the core code implementation, running the tests, and verifying the integration. Once the code is stable and all acceptance criteria are fully met, report back here. I will then invoke a specialized technical_writer subagent to perform a comprehensive audit and update ARCHITECTURE.md, README.md, and other documents to match the required mathematical and structural standards.

## Follow-up — 2026-06-02T18:50:54Z

Please resume the work on RideOn integration. The server recently crashed and restarted. Verify the current implementation status in the workspace, check which files have been modified, rerun the test suite to ensure health, and continue with the remaining milestones. Once the integration is complete, stable, and verified, please report back.
