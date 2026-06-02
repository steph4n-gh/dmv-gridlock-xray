# Developer Guide: Extending Feeds and Sensors in DMV Gridlock X-Ray

This developer guide details the process of extending the DMV Gridlock X-Ray engine (`engine.py`) with new dynamic feeds, sensors, or data streams. It outlines the step-by-step code implementation, the dynamic update of friction penalties, and the underlying mathematical formulations for scaling edge weights and solving the Graph Laplacian.

---

## 1. Adding a New Dynamic Feed or Sensor to `engine.py`

To integrate a new dynamic data feed or sensor, you must follow the asynchronous, non-blocking polling pattern established in the processing engine. The engine runs on `asyncio` and gathers all tasks concurrently.

### Step A: Define the Polling Task
Create a new asynchronous function in `engine.py`. This function should run in an infinite loop, fetch data from your API endpoint, parse the fields, and update the global `state` dictionary.

```python
async def poll_new_sensor_feed(tree, nodes_list):
    """
    Polls the new dynamic sensor endpoint, snaps geographic coordinates to graph nodes,
    and updates the system friction state.
    """
    url = "https://api.dmvtransit.gov/new_sensor/positions"
    # Set headers or authentication tokens as required
    headers = {"Authorization": f"Bearer {API_KEY_ENV_VAR}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                async with session.get(url, timeout=15.0) as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                        new_detections = {}
                        
                        for item in payload.get("data", []):
                            sensor_id = item.get("sensor_id")
                            lat = float(item.get("latitude", 0.0))
                            lon = float(item.get("longitude", 0.0))
                            congestion_level = float(item.get("congestion_index", 0.0))  # e.g., 0.0 to 1.0
                            
                            # Geographic Snapping fallback using the pre-built KDTree
                            if lat != 0.0 and lon != 0.0:
                                dist, idx = tree.query([lat, lon])
                                # Snapping threshold: e.g., 0.001 degrees (~111 meters)
                                if dist < 0.001:
                                    node_id = nodes_list[idx]
                                    new_detections[node_id] = congestion_level
                                    
                        # Atomically update global state
                        state['new_sensor_detections'] = new_detections
                        
            except Exception as e:
                print(f"⚠️ Error polling new sensor feed: {e}")
                
            # Adhere to polling frequency requirements (e.g., poll every 60 seconds)
            await asyncio.sleep(60.0)
```

### Step B: Register the Task in `main()`
Add the new async polling task to `asyncio.gather` inside the `main()` function of `engine.py`:

```python
async def main():
    init_logger()
    W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = load_static_topology()
    await asyncio.gather(
        poll_weather(),
        poll_incidents(tree),
        poll_bikeshare(tree),
        poll_gtfs_rt(),
        poll_rideon_gtfs_rt(tree, nodes_list),
        poll_alerts(route_to_stops),
        poll_vehicle_positions(),
        poll_metrorail_rt(),
        poll_rail_positions(tree, stops_info, nodes_list),
        poll_new_sensor_feed(tree, nodes_list),  # <-- New task registered here
        main_loop(W, nodes_list, node_to_idx, stops_info, tree, route_to_stops)
    )
```

---

## 2. Dynamic Weight and Friction Updates

Once data is stored in the global `state`, the core mathematical calculation loop (`main_loop`) must consume it to dynamically scale stop-level friction scores.

### Multiplicative Friction Penalty Formulation
Inside `main_loop` in `engine.py`, the friction vector $f \in \mathbb{R}^N$ is initialized to the baseline weather penalty in each cycle:

$$f_i = \text{weather\_penalty} \quad \forall i \in \{1, \dots, N\}$$

To inject penalties from your new sensor, update the friction scores multiplicatively:

```python
# Apply penalties from the new sensor feed inside the main_loop:
new_detections = state.get('new_sensor_detections', {})
for node_id, congestion_index in new_detections.items():
    if node_id in node_to_idx:
        idx = node_to_idx[node_id]
        
        # Scaling model: friction scales down as congestion index grows (0.0 to 1.0)
        # e.g., mapping congestion index to a penalty multiplier in [0.2, 1.0]
        penalty = 1.0 - (0.8 * congestion_index)
        f[idx] *= penalty
```

### Physical Clamping and numerical Stability
To prevent singularity collapses in the sparse eigensolver, you must enforce the physical clamp floor:

$$f_i = \max(f_i, 0.01) \quad \forall i \in \{1, \dots, N\}$$

```python
        # Enforce floor clamp
        f[idx] = max(f[idx], 0.01)
```

### Spatial Friction Diffusion (Graph Heat Kernel)
To simulate the back-pressure of queuing vehicles spilling onto adjacent roadways, apply a single-step Graph Laplacian heat equation after applying direct penalties. The friction value spreads to topological neighbors via:

$$f_{\text{smoothed}} = (1 - \alpha) f + \alpha D_{\text{mask}}^{-1} W_{\text{mask}} f$$

Where:
*   $\alpha \in [0, 1]$ is the diffusion rate (set as `ALPHA_DIFFUSION` = 0.20).
*   $W_{\text{mask}} \in \{0, 1\}^{N \times N}$ is the unweighted topological adjacency matrix.
*   $D_{\text{mask}}^{-1}$ is the inverse degree matrix of $W_{\text{mask}}$.

In Python, this is implemented using SciPy sparse matrix-vector multiplication:

```python
D_inv_W_f = D_inv * (W_mask @ f)
f = (1.0 - ALPHA_DIFFUSION) * f + ALPHA_DIFFUSION * D_inv_W_f
f = np.clip(f, 0.01, 1.0)  # Re-enforce stability limits [0.01, 1.0]
```

---

## 3. Mathematical Graph Laplacian & Solvers

The engine represents the transit grid as a live sparse matrix. When a developer modifies node counts or edge configurations, the underlying matrix sizes and solver steps scale accordingly.

### Symmetric Weight Scaling
The unweighted structural matrix $W_{\text{mask}}$ is scaled by the diagonal friction matrix $\text{diag}(f)$ to compute the weighted adjacency matrix $W$:

$$W = \text{diag}(f) \cdot W_{\text{mask}} \cdot \text{diag}(f)$$

For any edge $(i, j)$, this scales the connection strength symmetrically:

$$W_{ij} = f_i \cdot W_{\text{mask}, ij} \cdot f_j$$

If the friction score of either node collapses (approaching $0.01$), the joint edge weight drops, representing a physical structural bottleneck.

### The Graph Laplacian
The weighted Graph Laplacian $L$ is computed as the difference between the diagonal degree matrix $D$ and the weighted adjacency matrix $W$:

$$L = D - W$$

Where $D$ is defined by:

$$D_{ii} = \sum_{j=1}^{N} W_{ij}$$

The matrix $L$ is a symmetric, positive semi-definite matrix.

### Solving for Algebraic Connectivity
To find the algebraic connectivity ($\lambda_2$, the "Fiedler value"), we solve the sparse eigenvalue problem for the lowest eigenvalues:

$$L \vec{v} = \lambda \vec{v}$$

Since $L \vec{1} = 0$, the smallest eigenvalue is always $\lambda_1 = 0$. The second smallest eigenvalue $\lambda_2$ corresponds to the Fiedler value, and its eigenvector $\vec{v}_2$ represents the Fiedler vector.

Using SciPy's sparse eigensolver (`scipy.sparse.linalg.eigsh`), the system extracts the first three eigenvalues and eigenvectors:

```python
def spectral_analysis(W):
    n = W.shape[0]
    D_diag = np.array(W.sum(axis=1)).flatten()
    L = sp.diags(D_diag) - W
    try:
        # Requesting k=3 lowest eigenvalues (using shift-invert mode 'LM' around sigma=1e-5)
        evals, evecs = eigsh(L, k=3, which='LM', sigma=1e-5, tol=1e-2, maxiter=500)
        idx = np.argsort(evals)
        
        l2 = evals[idx[1]]
        v2 = evecs[:, idx[1]]
        spectral_gap = evals[idx[2]] - evals[idx[1]]
        
        return l2, v2, spectral_gap
    except Exception as e:
        # Fallback routines if solver fails to converge
        try:
            evals, evecs = eigsh(L, k=3, which='SM', tol=1e-1)
            idx = np.argsort(evals)
            return evals[idx[1]], evecs[:, idx[1]], evals[idx[2]] - evals[idx[1]]
        except:
            return 0.0, np.zeros(n), 0.0
```

### Network Topology Scaling and Edge Changes
When new nodes (stops) or edge types (routes) are permanently added to the system:
1.  **Node-to-Index Maps**: Update the static topology generation in `load_static_topology()` to append the new nodes, increasing the total vertex count to $N'$.
2.  **Matrix Resize**: The dimensions of $W_{\text{mask}}$ will automatically scale to $N' \times N'$. Because SciPy's sparse solvers support arbitrary sizes, the eigensolver will run seamlessly on the resized matrix.
3.  **Degree Recalculation**: The degree vector $D_{\text{diag}}$ and normalized centrality vector $C$ must be recomputed on startup, as their lengths scale to $N'$.
4.  **Vectorized Threshold Slices**: The pre-computed vectorized edge threshold array must be recomputed to align with the new adjacency indices:
    
    $$\text{Threshold}_{ij} = 0.4 \cdot (1.0 + \gamma \cdot (1.0 - \text{mean}(C_i, C_j)))$$
    
    where $\gamma = 0.50$ (`GAMMA_CENTRALITY`) and $C_i = D_i / \max(D)$.
