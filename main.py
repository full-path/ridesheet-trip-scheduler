# main.py
from flask import Request, jsonify
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List, Dict, Tuple


# ---------------------------
# Helpers: build instances & expand the matrix
# ---------------------------

def build_instances_for_requests(
    depot_base_idx: int,
    requests: List[Dict],
) -> Tuple[List[int], Dict[str, Tuple[int, int]], int]:
    """
    Creates one node instance per pickup and per dropoff (plus a single depot instance).
    Returns:
      - instance_to_base: list[int] mapping each node instance -> base location index
                          (index 0 is reserved for the depot instance)
      - request_node_indices: {request_id: (pickup_node_idx, dropoff_node_idx)}
      - depot_node_index: int (always 0 here)
    The resulting total node count N = 1 + 2 * len(requests).
    """
    instance_to_base: List[int] = []
    request_node_indices: Dict[str, Tuple[int, int]] = {}

    # Reserve index 0 for the depot instance
    depot_node_index = 0
    instance_to_base.append(depot_base_idx)
    next_idx = 1

    for i, r in enumerate(requests):
        rid = r.get("id", f"request_{i}")
        p_base = int(r["pickup_base"])
        d_base = int(r["dropoff_base"])

        p_idx = next_idx
        instance_to_base.append(p_base)
        next_idx += 1

        d_idx = next_idx
        instance_to_base.append(d_base)
        next_idx += 1

        request_node_indices[rid] = (p_idx, d_idx)

    return instance_to_base, request_node_indices, depot_node_index


def expand_time_matrix_from_base(
    base_matrix: List[List[int]],
    instance_to_base: List[int],
    same_place_travel_minutes: int = 0,
) -> List[List[int]]:
    """
    Expand an unduplicated base matrix (unique locations) into an instance-level matrix.
    For two different instances that share the same base location, set travel time to
    `same_place_travel_minutes` (default 0). Otherwise copy from base_matrix[base_i][base_j].
    """
    N = len(instance_to_base)
    out = [[0] * N for _ in range(N)]
    for i in range(N):
        bi = instance_to_base[i]
        for j in range(N):
            if i == j:
                out[i][j] = 0
            else:
                bj = instance_to_base[j]
                if bi == bj:
                    out[i][j] = int(same_place_travel_minutes)
                else:
                    out[i][j] = int(base_matrix[bi][bj])
    return out


# ---------------------------
# Validation
# ---------------------------

def _validate_input(body: dict):
    """
    Required top-level keys (new base-matrix schema):
      - base_time_matrix: square list[list[int]] (non-negative)
      - depot_base_index: int (index into base_time_matrix)
      - vehicles: list of objects:
          { id: str, time_window:[lo,hi], seat_capacity:int>=0, wc_capacity:int>=0 }
      - requests: list of objects:
          { id?:str, pickup_base:int, dropoff_base:int, seats?:int>=0, wheelchairs?:int>=0,
            pickup_tw?:[lo,hi], dropoff_tw?:[lo,hi], max_ride?:int>=0, penalty?:int>=0,
            pickup_service?:int>=0, dropoff_service?:int>=0 }
    Optional top-level keys:
      - same_place_travel_minutes:int>=0 (default 0)
      - default_pickup_service:int>=0 (default 3)
      - default_dropoff_service:int>=0 (default 3)
      - max_slack_minutes, horizon_minutes, solver_time_limit_sec,
        default_penalty, default_max_ride_minutes  (ints >= 0)
    """
    required = ["base_time_matrix", "depot_base_index", "vehicles", "requests"]
    for key in required:
        if key not in body:
            raise ValueError(f"Missing required key: '{key}'")

    # base_time_matrix
    bm = body["base_time_matrix"]
    if not isinstance(bm, list) or not all(isinstance(row, list) for row in bm):
        raise ValueError("'base_time_matrix' must be a list of lists.")
    L = len(bm)
    if L == 0:
        raise ValueError("'base_time_matrix' must not be empty.")
    if any(len(row) != L for row in bm):
        raise ValueError("'base_time_matrix' must be a square matrix.")
    if any((x is None or x < 0) for row in bm for x in row):
        raise ValueError("'base_time_matrix' must contain non-negative numbers only.")

    # depot_base_index
    depot_base = body["depot_base_index"]
    if not isinstance(depot_base, int) or not (0 <= depot_base < L):
        raise ValueError("'depot_base_index' is out of bounds for the given base_time_matrix.")

    # vehicles
    vehicles = body["vehicles"]
    if not isinstance(vehicles, list) or len(vehicles) == 0:
        raise ValueError("'vehicles' must be a non-empty list.")

    seen_ids = set()
    for i, v in enumerate(vehicles):
        if not isinstance(v, dict):
            raise ValueError(f"vehicles[{i}] must be an object.")
        vid = v.get("id")
        if not isinstance(vid, str) or not vid.strip():
            raise ValueError(f"vehicles[{i}].id must be a non-empty string.")
        if vid in seen_ids:
            raise ValueError(f"Duplicate vehicle id '{vid}'. Vehicle ids must be unique.")
        seen_ids.add(vid)

        tw = v.get("time_window")
        if (not isinstance(tw, list)) or len(tw) != 2 or tw[0] > tw[1]:
            raise ValueError(f"vehicles[{i}].time_window must be [lo, hi] with lo <= hi.")

        for key in ("seat_capacity", "wc_capacity"):
            cap = v.get(key)
            if not isinstance(cap, int) or cap < 0:
                raise ValueError(f"vehicles[{i}].{key} must be a non-negative integer.")

    # requests
    reqs = body["requests"]
    if not isinstance(reqs, list):
        raise ValueError("'requests' must be a list.")
    for i, r in enumerate(reqs):
        if not isinstance(r, dict):
            raise ValueError(f"requests[{i}] must be an object.")
        # base indices
        p_base, d_base = r.get("pickup_base"), r.get("dropoff_base")
        if p_base is None or d_base is None:
            raise ValueError(f"Request {i} is missing 'pickup_base' or 'dropoff_base'.")
        if not (isinstance(p_base, int) and isinstance(d_base, int)):
            raise ValueError(f"Request {i} 'pickup_base' and 'dropoff_base' must be integers.")
        if not (0 <= p_base < L and 0 <= d_base < L):
            raise ValueError(f"Request {i} base indexes out of bounds for base_time_matrix.")

        # time windows if present
        if "pickup_tw" in r:
            p_tw = r["pickup_tw"]
            if not (isinstance(p_tw, list) and len(p_tw) == 2 and p_tw[0] <= p_tw[1]):
                raise ValueError(f"Invalid 'pickup_tw' in request {i}. Must be [lo, hi] with lo <= hi.")
        if "dropoff_tw" in r:
            d_tw = r["dropoff_tw"]
            if not (isinstance(d_tw, list) and len(d_tw) == 2 and d_tw[0] <= d_tw[1]):
                raise ValueError(f"Invalid 'dropoff_tw' in request {i}. Must be [lo, hi] with lo <= hi.")

        # optional non-negative ints
        for key in ("seats", "wheelchairs", "max_ride", "penalty", "pickup_service", "dropoff_service"):
            if key in r and (not isinstance(r[key], int) or r[key] < 0):
                raise ValueError(f"Request {i} '{key}' must be a non-negative integer if provided.")


# ---------------------------
# Cloud Function
# ---------------------------

def solve_http(request: Request):
    """
    Entry point. Expects a base (unduplicated) travel-time matrix over unique locations,
    request pickup/dropoff indices into that base matrix, and vehicles. The function
    expands to an instance-level matrix (depot + 2*requests) for OR-Tools DARP.

    Input (example):
    {
      "base_time_matrix": [[0,12,15],[12,0,8],[15,8,0]],
      "depot_base_index": 0,
      "vehicles": [
        {"id":"Run-8AM-Blue","time_window":[480,1020],"seat_capacity":2,"wc_capacity":1}
      ],
      "requests": [
        {"id":"leg-1","pickup_base":1,"dropoff_base":2,"seats":1,"pickup_tw":[540,600]},
        {"id":"leg-2","pickup_base":2,"dropoff_base":1,"seats":1,"pickup_tw":[900,960]}
      ],
      "same_place_travel_minutes": 0,
      "default_pickup_service": 3,
      "default_dropoff_service": 3,
      "max_slack_minutes": 60,
      "horizon_minutes": 1440,
      "solver_time_limit_sec": 5,
      "default_penalty": 5000,
      "default_max_ride_minutes": 120
    }
    """
    body = request.get_json(force=True, silent=True) or {}

    try:
        # 1) Validate
        _validate_input(body)

        # 2) Parse base-level inputs
        base_tm: List[List[int]] = body["base_time_matrix"]
        depot_base_idx: int = int(body["depot_base_index"])
        reqs: List[Dict] = body["requests"]
        vehicles: List[Dict] = body["vehicles"]

        same_place_travel = int(body.get("same_place_travel_minutes", 0))
        default_pickup_service = int(body.get("default_pickup_service", 3))
        default_dropoff_service = int(body.get("default_dropoff_service", 3))

        max_slack = int(body.get("max_slack_minutes", 60))
        horizon = int(body.get("horizon_minutes", 24 * 60))
        time_limit_sec = int(body.get("solver_time_limit_sec", 5))
        default_penalty = int(body.get("default_penalty", 5000))
        default_max_ride = int(body.get("default_max_ride_minutes", 120))

        # 3) Build instance set (depot + one pickup & one dropoff per request)
        instance_to_base, request_node_indices, depot_node_index = build_instances_for_requests(
            depot_base_idx=depot_base_idx,
            requests=reqs,
        )
        n_loc = len(instance_to_base)        # total OR-Tools nodes
        n_veh = len(vehicles)

        # 4) Expand time matrix to instance level
        tm = expand_time_matrix_from_base(
            base_matrix=base_tm,
            instance_to_base=instance_to_base,
            same_place_travel_minutes=same_place_travel
        )

        # 5) Build per-instance service_time
        #    depot assumed 0; pickups/dropoffs can use per-request override or defaults
        svc = [0] * n_loc
        for i, r in enumerate(reqs):
            rid = r.get("id", f"request_{i}")
            p_idx, d_idx = request_node_indices[rid]
            svc[p_idx] = int(r.get("pickup_service", default_pickup_service))
            svc[d_idx] = int(r.get("dropoff_service", default_dropoff_service))

        # 6) Normalize vehicles → arrays OR-Tools expects
        v_tw   = [v["time_window"]   for v in vehicles]
        v_seat = [v["seat_capacity"] for v in vehicles]
        v_wc   = [v["wc_capacity"]   for v in vehicles]
        vehicle_ids = [v["id"] for v in vehicles]

        # 7) Convert requests to instance indexes & collect demands/windows/params
        #    (we'll keep the original req list but access p/d node indices via request_node_indices)
    except (ValueError, TypeError, KeyError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    # ---------------------------
    # OR-Tools model
    # ---------------------------
    manager = pywrapcp.RoutingIndexManager(n_loc, n_veh, depot_node_index)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback: add service time at "from" node so CumulVar(node) = departure time
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(tm[i][j] + (svc[i] if i != depot_node_index else 0))

    tcb_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tcb_idx)

    # Time dimension (waiting allowed up to max_slack)
    routing.AddDimension(
        tcb_idx,
        max_slack,     # waiting cap per node
        horizon,       # route horizon
        False,         # don't fix start cumul to 0
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Vehicle shift windows at Start/End
    for v in range(n_veh):
        s, e = routing.Start(v), routing.End(v)
        lo, hi = v_tw[v]
        time_dim.CumulVar(s).SetRange(int(lo), int(hi))
        time_dim.CumulVar(e).SetRange(int(lo), int(hi))

    # Capacity dimensions: Seats and Wheelchairs
    seats_dem = [0] * n_loc
    wc_dem = [0] * n_loc
    for i, r in enumerate(reqs):
        rid = r.get("id", f"request_{i}")
        p_idx, d_idx = request_node_indices[rid]
        seats_dem[p_idx] += int(r.get("seats", 1))
        seats_dem[d_idx] -= int(r.get("seats", 1))
        wc_dem[p_idx]    += int(r.get("wheelchairs", 0))
        wc_dem[d_idx]    -= int(r.get("wheelchairs", 0))

    seats_cb = routing.RegisterUnaryTransitCallback(lambda idx: seats_dem[manager.IndexToNode(idx)])
    wc_cb    = routing.RegisterUnaryTransitCallback(lambda idx: wc_dem[manager.IndexToNode(idx)])
    routing.AddDimensionWithVehicleCapacity(seats_cb, 0, v_seat, True, "Seats")
    routing.AddDimensionWithVehicleCapacity(wc_cb, 0, v_wc, True, "Wheelchairs")

    # Pairing, time windows, max ride, and drop penalties per request
    for i, r in enumerate(reqs):
        rid = r.get("id", f"request_{i}")
        p_node, d_node = request_node_indices[rid]
        pI, dI = manager.NodeToIndex(p_node), manager.NodeToIndex(d_node)

        routing.AddPickupAndDelivery(pI, dI)

        if "pickup_tw" in r:
            lo, hi = r["pickup_tw"]
            time_dim.CumulVar(pI).SetRange(int(lo), int(hi))
        if "dropoff_tw" in r:
            lo, hi = r["dropoff_tw"]
            time_dim.CumulVar(dI).SetRange(int(lo), int(hi))

        max_ride = int(r.get("max_ride", default_max_ride))
        penalty = int(r.get("penalty", default_penalty))
        act_p = routing.ActiveVar(pI)
        act_d = routing.ActiveVar(dI)
        # Make the pair optional as a unit
        routing.solver().Add(act_p == act_d)
        routing.AddDisjunction([pI], penalty)
        # Only enforce if the pair is active
        routing.solver().Add(time_dim.CumulVar(pI) <= time_dim.CumulVar(dI)).only_enforce_if(act_p)
        routing.solver().Add(routing.VehicleVar(pI) == routing.VehicleVar(dI)).only_enforce_if(act_p)
        routing.solver().Add(time_dim.CumulVar(dI) - svc[d_node] - time_dim.CumulVar(pI) <= max_ride).only_enforce_if(act_p)

    # Search
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_sec
    params.log_search = True

    sol = routing.SolveWithParameters(params)
    status = routing.status()
    status_map = {
        0: "ROUTING_NOT_SOLVED",
        1: "ROUTING_SUCCESS",
        2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
        3: "ROUTING_FAIL",
        4: "ROUTING_FAIL_TIMEOUT",
        5: "ROUTING_INVALID",
        6: "ROUTING_INFEASIBLE",
    }
    status_msg = status_map.get(status, f"Unknown status code {status}")

    if not sol:
        return jsonify({"solution_found": False, "status": status_msg}), 200

    # ---------------------------
    # Extract solution
    # ---------------------------
    out = {
        "solution_found": True,
        "status": status_msg,
        "vehicles": [],
        "request_assignments": {},
        "unserved_requests": []
    }

    # Multi-valued node metadata: which request IDs pick up / drop at each instance node
    node_meta = {depot_node_index: {"pickups": [], "dropoffs": []}}
    for i, r in enumerate(reqs):
        rid = r.get("id", f"request_{i}")
        p_node, d_node = request_node_indices[rid]
        node_meta.setdefault(p_node, {"pickups": [], "dropoffs": []})["pickups"].append(rid)
        node_meta.setdefault(d_node, {"pickups": [], "dropoffs": []})["dropoffs"].append(rid)

    def cumul(idx): return sol.Value(time_dim.CumulVar(idx))

    # Per-vehicle routes
    for v in range(n_veh):
        idx = routing.Start(v)
        route = []
        served_rids_for_vehicle = set()

        # First node (depot start)
        node = manager.IndexToNode(idx)
        depart = cumul(idx)
        arrive = depart - (svc[node] if node != depot_node_index else 0)
        meta = node_meta.get(node, {"pickups": [], "dropoffs": []})
        served_rids_for_vehicle.update(meta["pickups"])
        served_rids_for_vehicle.update(meta["dropoffs"])
        route.append({
            "node": node,
            "base_loc_index": instance_to_base[node],
            "pickups": meta["pickups"],      # list of request IDs
            "dropoffs": meta["dropoffs"],    # list of request IDs
            "arrival_minute": int(arrive),
            "departure_minute": int(depart),
            "wait_minutes": 0
        })

        # Walk the chain
        while not routing.IsEnd(idx):
            prev_idx = idx
            idx = sol.Value(routing.NextVar(idx))
            prev_node = manager.IndexToNode(prev_idx)
            node = manager.IndexToNode(idx)

            travel = tm[prev_node][node]
            depart_prev = cumul(prev_idx)
            depart_cur  = cumul(idx)
            arrive_cur  = depart_cur - (svc[node] if node != depot_node_index else 0)
            min_arrival_if_no_wait = depart_prev + travel
            wait = max(0, arrive_cur - min_arrival_if_no_wait)

            meta = node_meta.get(node, {"pickups": [], "dropoffs": []})
            served_rids_for_vehicle.update(meta["pickups"])
            served_rids_for_vehicle.update(meta["dropoffs"])
            route.append({
                "node": node,
                "base_loc_index": instance_to_base[node],
                "pickups": meta["pickups"],
                "dropoffs": meta["dropoffs"],
                "arrival_minute": int(arrive_cur),
                "departure_minute": int(depart_cur),
                "wait_minutes": int(wait)
            })

        out["vehicles"].append({
            "vehicle": v,                     # numeric solver index
            "vehicle_id": vehicle_ids[v],     # stable external ID from input
            "route": route,
            "requests": sorted(served_rids_for_vehicle)
        })

    # Direct request -> vehicle mapping with timestamps
    for i, r in enumerate(reqs):
        rid = r.get("id", f"request_{i}")
        p_node, d_node = request_node_indices[rid]
        pI, dI = manager.NodeToIndex(p_node), manager.NodeToIndex(d_node)

        if sol.Value(routing.ActiveVar(pI)) == 1:
            v = sol.Value(routing.VehicleVar(pI))
            p_depart = cumul(pI); d_depart = cumul(dI)
            p_arrive = p_depart - (svc[p_node] if p_node != depot_node_index else 0)
            d_arrive = d_depart - (svc[d_node] if d_node != depot_node_index else 0)

            out["request_assignments"][rid] = {
                "vehicle": v,
                "vehicle_id": vehicle_ids[v],
                "pickup_node": p_node,
                "pickup_base_loc_index": instance_to_base[p_node],
                "dropoff_node": d_node,
                "dropoff_base_loc_index": instance_to_base[d_node],
                "pickup":  {"arrival_minute": int(p_arrive), "departure_minute": int(p_depart)},
                "dropoff": {"arrival_minute": int(d_arrive), "departure_minute": int(d_depart)}
            }
        else:
            out["unserved_requests"].append(rid)

    return jsonify(out), 200
