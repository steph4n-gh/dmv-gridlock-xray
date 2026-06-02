# DMV Gridlock X-Ray Engine Documentation

## Core Concept
The DMV Gridlock X-Ray Engine transforms the Washington D.C. metropolitan transit network (WMATA and Montgomery County RideOn) into a live mathematical topology. Instead of simply plotting vehicle markers on a static canvas, it models the entire transit grid as a sparse mathematical matrix. By calculating the algebraic connectivity ($\lambda_2$, the "Fiedler value") of this matrix every 30 seconds, the engine detects and visually maps structural gridlock before human dispatchers issue official alerts.

---

## The Friction Array ($f$)
Every single stop/station node $i \in V$ in the network starts with a baseline friction score $f_i$ of $1.0$ (representing perfect traffic flow). 

As the engine polls multimodal data streams, it dynamically multiplies this baseline by various penalties. A lower score indicates greater traffic friction, clamped by the numerical stability floor:
$$f_i = \max(f_i, 0.01)$$

### Data Sources & Penalties
1. **Weather (Open-Meteo)**
   - *Data*: Live precipitation codes.
   - *Penalty*: $\text{WEATHER\_PENALTY} = 0.85$ (Global applied to all nodes if raining/snowing).
   
2. **Municipal Incidents (DC/MD Police CAD with Crowdsourced Waze Integration)**
   - *Data*: GeoJSON crash and road closure feeds snapped to network nodes via KDTree lookup within $\theta_{\text{snap}} = 0.008^{\circ}$ (approx. $880\text{ meters}$).
   - *Penalty*: $\text{INCIDENT\_PENALTY} = 0.05$ (Clamps node friction directly to the structural gridlock floor).

3. **Live WMATA Metrobus Velocity (GTFS-RT VehiclePositions)**
   - *Data*: Real-time GPS speed of active WMATA buses.
   - *Penalty*:
     - Speed $< 15\text{ mph}$ (or $< 6.7056\text{ m/s}$): $0.6$ (Heavy Traffic)
     - Speed $< 5\text{ mph}$ (or $< 2.2352\text{ m/s}$): $0.3$ (Severe Gridlock)

4. **Live RideOn Bus Velocity (JSON REST VehiclePositions & TripUpdates)**
   - *Data*: Real-time GPS speed, bearing, and route positions parsed from JSON REST feeds.
   - *Snapping*: Ingested coordinates are processed using a KDTree snapping fallback algorithm to assign a nearby stop node ID if the telemetry payload's `StopId` field is missing:
     $$\text{dist}_{\text{deg}}(x_{\text{bus}}, x_j) < 0.001^{\circ}$$
   - *Penalty*:
     - Speed $< 15\text{ mph}$ (or $< 6.7056\text{ m/s}$): $0.6$ (Heavy Traffic)
     - Speed $< 5\text{ mph}$ (or $< 2.2352\text{ m/s}$): $0.3$ (Severe Gridlock)

5. **Metrorail Surface Surges (WMATA rail-gtfsrt-alerts)**
   - *Data*: General text alerts for train delays (e.g., "Red Line Single Tracking").
   - *Penalty*: $\text{RAIL\_SURGE\_PENALTY} = 0.70$ applied to the top 2% most highly connected bus hubs in the city (simulating stranded passengers flooding the surface).

6. **Metrorail Precision Ground Zero (WMATA rail-gtfsrt-vehiclepositions)**
   - *Data*: Live GPS of underground trains.
   - *Penalty*: $\text{GROUND\_ZERO\_PENALTY} = 0.60$ applied to surface bus stops located within $500\text{ meters}$ of a train that has physically stopped (status `STOPPED_AT`) during an active rail alert.

---

## Centrality-Adjusted Edge Fracture Thresholds
The joint weight representing the flow capacity between Node $i$ and Node $j$ is scaled as:
$$\text{Weight}_{ij} = f_i \cdot f_j$$

An edge $(u, v)$ is flagged as "fractured" if its joint weight drops below the centrality-adjusted edge threshold:
$$\text{Threshold}_{ij} = 0.4 \cdot (1.0 + \gamma \cdot (1.0 - \text{mean}(C_i, C_j)))$$

Where:
- $\gamma = 0.50$ (`GAMMA_CENTRALITY`) is the degree-centrality scaling factor.
- $C_k = D_k / \max(D)$ represents the normalized degree centrality of node $k$.
- $\text{mean}(C_i, C_j) = \frac{C_i + C_j}{2}$ is the average normalized centrality of the connecting nodes.

### Mathematical Proof of Dual Sensitivity Boundaries
This formulation guarantees that the fracture detection threshold dynamically scales between $0.40$ and $0.60$ based on local topological redundancy:

#### 1. Highly Connected Downtown Hubs (High Centrality Limit)
For downtown transit hubs with massive node degree connectivity (e.g., Union Station, Metro Center), $C_i \approx 1.0$ and $C_j \approx 1.0$, yielding $\text{mean}(C_i, C_j) \approx 1.0$.
$$\text{Threshold}_{\text{urban}} \approx 0.4 \cdot (1.0 + 0.5 \cdot (1.0 - 1.0)) = 0.40$$

*Rationale*: High topological routing redundancy allows downtown grids to absorb minor delays without immediate systemic failures. Keeping the threshold at the conservative $0.40$ baseline filters out urban background noise and prevents false alarms.

#### 2. Isolated Suburban Corridors (Low Centrality Limit)
For sparse suburban corridors with minimal connections (e.g., single-highway corridors, isolated commuter stops), $C_i \approx 0.0$ and $C_j \approx 0.0$, yielding $\text{mean}(C_i, C_j) \approx 0.0$.
$$\text{Threshold}_{\text{suburban}} \approx 0.4 \cdot (1.0 + 0.5 \cdot (1.0 - 0.0)) = 0.60$$

*Rationale*: Suburban transit corridors lack topological redundancy. A single delayed vehicle or minor local blockage has immediate and catastrophic upstream/downstream impacts on the entire sub-grid. Raising the threshold to $0.60$ increases detection sensitivity to capture these critical fractures instantly.

---

## The Adjudication Engine & Scoreboard
Because WMATA's official Service Alerts are decoupled from the friction math, the engine is forced into a competitive match against WMATA's human dispatchers. 

When a structural fracture opens on the map, the **Canary Protocol** automatically locks onto the GPS feed of a physical bus approaching that tear. The engine tracks when the bus hits the gridlock ($T_{\text{physical}}$) and waits to see if/when WMATA issues a service bulletin ($T_{\text{wmata}}$). The delta is logged to the persistent `prediction_scoreboard.csv` ledger.
