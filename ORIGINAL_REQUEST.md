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

## Follow-up — 2026-06-02T20:00:31Z

The goal of this task is to perform a final documentation pass and repository cleanup for the DMV Gridlock X-Ray. This includes copying and embedding visual map screenshots, clarifying setup requirements, providing an extension guide for adding feeds and recalculating weights, documenting the research nature of the project, and adding an MIT license.

Working directory: /Volumes/Storage/wmataworkspace/bustracker
Integrity mode: development

## Requirements

### R1. Copy and Embed 2D/3D Map Screenshots
- Create a `docs/screenshots/` directory in the repository.
- Copy at least three screenshots (e.g. the 2D dashboard/Mapbox map `media__1780429164336.png`, the top-down road/rail network overview `media__1780429193678.jpg`, and the 3D grid `media__1780429164260.png`) from the App Data / Brain directory `/Users/sarrington/.gemini/antigravity/brain/d7e0ce51-4a99-4499-ae92-40a5230c9739/` to `docs/screenshots/`.
- Embed these screenshots dynamically in `README.md` to make the repository visually engaging.

### R2. Clarify Setup Requirements
- Explicitly detail what configurations and keys the user must provide (specifically, environment variables `WMATA_API_KEY`, `RIDEON_API_KEY`, and `RIDEON_CLIENT_ID`).
- Document how the GTFS static files are handled: the system automatically checks for the static files (in `gtfs/wmata/` and `gtfs/rideon/`) at startup, and downloads/extracts them if they are missing. Clarify that no manual GTFS downloading is required.

### R3. Extension and Weight Recalculation Guide
- Add a new section in `ARCHITECTURE.md` (or create a dedicated `docs/extending_feeds.md` developer guide) that details:
  1. How to add a new dynamic feed or sensor to `engine.py`.
  2. How weights/friction penalties should be updated dynamically based on new additions.
  3. The mathematical implementation details for scaling edge weights $W_{ij} = f_i \cdot W_{ij} \cdot f_j$ and solving the Graph Laplacian $L = D - W$ when nodes or edge types change.

### R4. Project Research Status & MIT License
- Add a prominent section in `README.md` noting that this is an experimental research project exploring Spectral Graph Theory applied to real-time public transit networks.
- Create a `LICENSE` file in the repository root containing the standard MIT License.
- Add a "License" section at the bottom of `README.md` referencing the MIT License.

## Acceptance Criteria

### Documentation Updates
- [ ] Visual screenshots (including the 2D Mapbox router view) are present in the `docs/screenshots/` directory and successfully embedded in `README.md`.
- [ ] Setup guide in `README.md` clearly states that static GTFS downloads are managed automatically by the application on startup, requiring only the environment API keys.
- [ ] Extending and weights recalculation guide is present and mathematically accurate in `ARCHITECTURE.md` or a new developer guide.
- [ ] A valid `LICENSE` file (MIT) is present in the repository root and linked from `README.md`.
- [ ] All 67 unit and integration tests compile and pass.
