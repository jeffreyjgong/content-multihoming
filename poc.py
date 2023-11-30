import numpy as np
from typing import List, Dict, Tuple, Set

# Example structure for DIDs (Data Identifiers) and RSEs (Rucio Storage Elements)
# DIDs will be represented as strings and RSEs as integers for simplicity
DID = str
RSE = int

# Example cost function, modeled as a concave function (simplified version)
def cost_function(size: int) -> float:
    return 1 / np.sqrt(size) if size > 0 else 0

# Function to convert a DID into a vector representation
# This is a placeholder function and will need actual logic based on the problem specifics
def vAsVector(did: DID, feasible_rses: Dict[RSE, Set[DID]]) -> np.ndarray:
    # Example: Vector length equals number of RSEs, with cost values if DID is feasible for the RSE
    vec = np.array([cost_function(1) if did in feasible_rses[rse] else 0 for rse in feasible_rses])
    return vec

# Function to identify hyperplanes
def identify_hyperplanes(dids: List[DID], feasible_rses: Dict[RSE, Set[DID]]) -> List[np.ndarray]:
    hyperplanes = []
    for did in dids:
        vVec = vAsVector(did, feasible_rses)
        for k in feasible_rses:
            for j in feasible_rses:
                if k != j and did in feasible_rses[k] and did in feasible_rses[j]:
                    e_k = np.zeros(len(feasible_rses))
                    e_j = np.zeros(len(feasible_rses))
                    e_k[k], e_j[j] = 1, 1
                    hpCandidate = np.cross(vVec, e_k - e_j)
                    hpCandidate = hpCandidate / np.linalg.norm(hpCandidate)  # Normalize
                    if not any(np.array_equal(hpCandidate, hp) for hp in hyperplanes):
                        hyperplanes.append(hpCandidate)
    return hyperplanes

# Example usage
print("Starting hyperplane identification!")
dids = ["DID1", "DID2", "DID3"]
feasible_rses = {0: {"DID1", "DID2"}, 1: {"DID2", "DID3"}, 2: {"DID1", "DID3"}}
hyperplanes = identify_hyperplanes(dids, feasible_rses)
print(hyperplanes) # Displaying the identified hyperplanes

print("Starting interior point calculation!")

# BLACKBOXED
# interior_points = computePs(hyperplanes)

print("Starting extremal assignment evaluation!")

def compute_extremal_assignments(dids: List[DID], feasible_rses: Dict[RSE, Set[DID]], interior_points: List[np.ndarray]) -> Dict[DID, RSE]:
    """
    Compute the extremal assignments for each DID based on the interior points and feasible RSEs.
    """
    optimal_assignment = {}

    for P in interior_points:
        current_optimal_assignment = {}
        for did in dids:
            optimal_rse = None
            optimal_cost = float('inf')
            for rse in feasible_rses:
                if did in feasible_rses[rse]:
                    vVec = vAsVector(did, feasible_rses)
                    e_k = np.zeros(len(feasible_rses))
                    e_k[rse] = 1
                    cost = np.dot(P, np.cross(vVec, e_k))
                    if cost < optimal_cost:
                        optimal_cost = cost
                        optimal_rse = rse
            current_optimal_assignment[did] = optimal_rse

        # Compare with the current optimal assignment and update if better
        if not optimal_assignment or sum(np.dot(P, vAsVector(did, feasible_rses)) for did in dids) < sum(np.dot(P, vAsVector(did, feasible_rses)) for did in optimal_assignment):
            optimal_assignment = current_optimal_assignment

    return optimal_assignment

# Placeholder for interior points from Step 2 (blackboxed)
interior_points = [np.random.rand(len(feasible_rses)) for _ in range(3)]  # Example interior points

# Compute extremal assignments
optimal_assignments = compute_extremal_assignments(dids, feasible_rses, interior_points)
print(optimal_assignments)  # Displaying the optimal assignments