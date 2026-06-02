# Porting Guide: Adapting DMV Gridlock X-Ray to Other Transit Systems

This guide explains how to adapt the DMV Gridlock X-Ray engine and visualization dashboard to other metropolitan transit networks around the United States (for example, transitioning from the Washington D.C. system to the New York City MTA). 

Because the engine is fundamentally driven by standard GTFS (static) and GTFS-Realtime (dynamic) protocols, the core mathematical Graph Laplacian logic remains identical. However, geographical bounds, snapping parameters, auxiliary API feeds, and structural classifications must be calibrated for the target metropolitan area.

---

## 1. Step-by-Step Adaption Blueprint

To port the system to a new city (e.g., NYC), follow these 6 sequential integration steps:

```
+-------------------------------------------------------------------+
| STEP 1: Define Geography & Viewport Parameters (Bounds, RBFs)    |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| STEP 2: Swap Static GTFS Files & Compile Adjacency Mask (W_mask)  |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| STEP 3: Bind Real-Time GTFS-RT APIs (Subway & Bus Feeds)          |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| STEP 4: Calibrate Spatial Snapping and Transfer Search Thresholds |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| STEP 5: Re-map Multimodal Canary Feeds (Bikeshare, Weather, CAD)   |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| STEP 6: Classify Subway Elevational Offsets & Tunnels             |
+-------------------------------------------------------------------+
```

---

## 2. Technical Modifications & Code Locations

### Step 1: Geography and Viewport Parameters
You must update the geographic coordinate boundaries and center-points in both the backend (`server.py`) and the frontend (`index.html`).

#### A. Frontend Viewports (`index.html`)
Update the default Mapbox map initialization to center on the target city (e.g., centering on Manhattan instead of Washington D.C.):
```javascript
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/dark-v10',
    center: [-73.9857, 40.7484], // NYC Center (longitude, latitude)
    zoom: 11.5,
    pitch: 45,
    bearing: -15
});
```

#### B. Terrain Gaussian Radial Basis Functions (`index.html`)
The continuous 3D Topographical View (`3D_TOPO`) uses $K$ multivariate Gaussian Radial Basis Functions (RBFs) to interpolate elevations on the client-side. Swap the D.C. topographical anchors for the target city's elevations (e.g., Manhattan ridges, East/Hudson River beds, and Brooklyn Heights):
$$h_{\text{topo}}(\phi, \lambda) = h_{\text{base}} + \sum_{k=1}^K A_k \exp\left( -\frac{(\phi - \phi_k)^2 + (\lambda - \lambda_k)^2}{2 \sigma_k^2} \right)$$

Configure the new anchors in the Javascript RBF evaluator:
```javascript
const topoAnchors = [
    { name: "Manhattan Midtown Ridge", lat: 40.7580, lon: -73.9850, A: 25.0, sigma: 0.015 },
    { name: "Brooklyn Heights", lat: 40.6920, lon: -73.9980, A: 35.0, sigma: 0.012 },
    { name: "Hudson River Valley", lat: 40.7480, lon: -74.0200, A: -30.0, sigma: 0.008 }
];
```

---

### Step 2: Static GTFS Swapping & Graph Construction
The structural topology of the grid is built from static GTFS files.

1.  **Clear Caches**: Remove existing WMATA/RideOn files in `gtfs/wmata/`, `gtfs/rideon/`, and delete the compiled state `network_state.npz`.
2.  **Add New Feeds**: Place the target agency's static GTFS text files (such as MTA Subways `google_transit.zip` and MTA Bus GTFS files) in `gtfs/mta_subway/` and `gtfs/mta_bus/`.
3.  **Update Loader Function (`engine.py`)**: Update the loader `load_static_topology()` to parse the new file paths.
4.  **Operator Prefixing**: Ensure all node IDs are prefixed (e.g., `mta_subway_` and `mta_bus_`) to keep separate operators isolated in the unified CSR sparse matrix.

---

### Step 3: Real-Time GTFS-RT Ingestion
Update the real-time API polling threads in `engine.py` to point to the new city's GTFS-RT streams.

*   **API Authentication**: MTA and other transit agencies often require authorization headers or custom API key fields. Implement these in `poll_gtfs_rt()` using environment variables.
*   **MTA Subway Feed Specifics**: Unlike WMATA, which groups rail positions in a single Protobuf stream, the MTA publishes separate GTFS-RT feeds for different lines (e.g., lines 1, 2, 3 on one feed, A, C, E on another). Update `poll_rail_positions()` to gather and aggregate from multiple URLs using `asyncio.gather`.

---

### Step 4: Calibration of Snapping Boundaries
The density of NYC's transit grid is significantly higher than Washington D.C.'s. You must scale the geometric snapping thresholds to prevent coordinate snapping collisions.

#### A. Stop Snapping Snuffer (`engine.py` & `server.py`)
Snapping geographic incident points $x_{\text{inc}}$ to stop nodes $x_i$ uses a KDTree search. Calibrate the snapping threshold $\theta_{\text{snap}}$ to match local node density:
$$\text{dist}(x_{\text{inc}}, x_i) < \theta_{\text{snap}}$$

*   *DC Baseline*: $\theta_{\text{snap}} = 0.008^{\circ}$ (approx. $880\text{m}$).
*   *NYC Calibration*: Reduce to $\theta_{\text{snap}} = 0.003^{\circ}$ (approx. $330\text{m}$) in high-density areas (like Manhattan) to prevent crashes on one block from clamping the friction of stops on adjacent avenues.

#### B. Walking Transfer Bounds (`server.py`)
During the construction of the augmented graph $G_{\text{aug}}$ used by the Dijkstra pathfinder, virtual walking transfer edges are created for adjacent stops.
*   *DC Baseline*: Snaps stops within $150.0\text{ meters}$.
*   *NYC Calibration*: Increase or decrease depending on the transit layout. For MTA subway stations with long underground passageways (e.g. Times Square to Port Authority), snap transfer boundaries must be set to $250.0\text{ meters}$ to allow the pathfinder to correctly resolve multi-operator transfers.

---

### Step 5: Auxiliary Multimodal Feeds
To maintain the integrity of the Multimodal Synergy Lab (recalculating the Panic Shift Index, Weather Drag, etc.), swap out the local auxiliary feeds:

1.  **Bikeshare GBFS**: Point to **Citi Bike** GBFS endpoints (`https://gbfs.citibikenyc.com/gbfs/en/station_information.json` and `station_status.json`).
2.  **Municipal Incidents**: Connect to the **NYC OpenData / NYCDOT Crash & Incident API** REST endpoint, snapping incident points using the updated KDTree search.
3.  **Weather Coordinates**: Update `poll_weather()` to query Open-Meteo at NYC coordinates (`latitude=40.7128`, `longitude=-74.0060`).

---

### Step 6: Subway Station Classifications
The 3D visualization snaps subway tunnels and trains to specific elevations. Add the new subway station classifications to `get_station_class()` in `server.py` and `index.html`:

*   **Deep Underground ($\Delta h = -45.0\text{m}$)**: For stations deep under bedrock or passing under rivers (e.g., 191st St on 1 Line, Clark St, Grand Central deep platforms).
*   **Standard Underground ($\Delta h = -20.0\text{m}$)**: Standard cut-and-cover subway platforms (e.g., standard Manhattan station platforms).
*   **Elevated Guideways ($\Delta h = +12.0\text{m}$)**: Elevated line tracks (e.g., Queensboro Plaza, Smith-9th Sts, elevated outer borough lines).
*   **At-grade Ballast ($\Delta h = 0.0\text{m}$)**: Balast tracks at ground level (e.g., Brighton Beach surface sections).

---

## 3. Recalculating Graph Laplacian Weights
When the new static topology is fused, the unweighted adjacency mask $W_{\text{mask}} \in \{0,1\}^{N \times N}$ is populated. The engine dynamically calculates the Graph Laplacian matrix $L$ using the updated node friction vector $f \in \mathbb{R}^N$:

$$W = \text{diag}(f) \cdot W_{\text{mask}} \cdot \text{diag}(f) \quad \implies \quad W_{ij} = f_i \cdot W_{\text{mask}, ij} \cdot f_j$$
$$L = D - W \quad \text{where} \quad D_{ii} = \sum_{j} W_{ij}$$

The new node degree vector $D$ and degree centrality vector $C_i = D_i / \max(D)$ scale automatically to the size of the new network $N$. The vectorized centrality-adjusted edge thresholds are automatically recompiled on startup:
$$\text{Threshold}_{ij} = 0.4 \cdot (1.0 + \gamma \cdot (1.0 - \text{mean}(C_i, C_j)))$$

This guarantees that the bisection boundary solver and Dijkstra pathfinder function properly on the new grid with zero manual matrix adjustments required.
