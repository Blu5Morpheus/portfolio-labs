import numpy as np

def check_positivity(u, v, threshold, curved=False):
    """
    Check if a point (u, v) satisfies the toy positive geometry constraints.
    
    Constraints:
    1. u > 0, v > 0 (First Quadrant)
    2. 1 - u - v > 0 (Simplex / Triangle)
    3. (Optional) u*v > threshold or similar curved constraint for 'higher loop' feeling.
    
    Returns:
        bool: True if allowed (positive), False otherwise.
    """
    # Basic Simplex: u > 0, v > 0, u + v < 1
    if u < 0 or v < 0:
        return False
    if u + v > 1:
        return False
        
    if curved:
        # Toy curved constraint: u*v > threshold
        # Or let's make it more interesting: u(1-u) + v(1-v) > threshold?
        # Let's use u*v > k * 0.1 to keep it scale-appropriate
        # Actually spec said: u*v > k
        if u * v < threshold:
            return False
            
    return True

def sample_kinematics(n_samples, threshold, curved=False):
    """
    Sample N points in the unit square [0,1]x[0,1] and score them.
    
    Returns:
        points (Nx2 array)
        scores (N array): 1.0 for allowed, 0.0 for forbidden (binary for now, or graded)
    """
    # Uniform sampling
    pts = np.random.rand(n_samples, 2)
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        u, v = pts[i]
        allowed = check_positivity(u, v, threshold, curved)
        if allowed:
            # Simple binary score for visualization
            scores[i] = 1.0
        else:
            # 0.2 for "dim" but visible
            scores[i] = 0.1
            
    return pts, scores

def calculate_boundary_distance(u, v, threshold, curved=False):
    """
    Toy distance score.
    Real distance to intersection of manifolds is hard.
    We'll return a score based on 'min margin' to any constraint.
    """
    # Margins (positive = inside)
    m1 = u
    m2 = v
    m3 = 1 - u - v
    
    margins = [m1, m2, m3]
    
    if curved:
        m4 = u*v - threshold
        margins.append(m4)
        
    min_margin = min(margins)
    
    # Map margin to score [0, 1]
    # If negative (forbidden), 0.
    # If positive, scale it. Max margin in simplex is roughly 0.3.
    if min_margin < 0:
        return 0.0
    
    score = min_margin / 0.3 # Scale factor
    return min(score, 1.0)

def compute_volume_form(vertices, probe):
    """
    Computes the canonical form Omega for a polygon.
    Omega = Sum_i  dlog(Z_i, Z_{i+1}, Y)
    
    For a 2D polygon with vertices Z, relative to probe Y.
    This is a heuristic scalar value representing the 'magnitude' of the form at Y.
    """
    # Simply 1/distance_to_boundary for visualization
    # Real form is a differential form. Use distance score as proxy magnitude.
    
    # Check if inside
    # If outside, 0.
    
    # We can reuse calculate_boundary_distance logic or do winding number.
    # For this toy model, let's just make it proportional to 1 / (min_dist + epsilon)
    
    # We need to map vertices to the u,v logic or checking geometry.
    # The Amplituhedron app uses 'calculate_boundary_distance' for the u,v space.
    # But for the Polygon visualizer, it passes vertices.
    
    # Let's implementation a simple 'Inverse distance product' to all edges
    
    # Edges
    dist_prod = 1.0
    
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i+1)%len(vertices)]
        
        # Distance from probe to line p1-p2
        # Line: ax + by + c = 0
        # Normal vector n = (-(y2-y1), x2-x1)
        
        normal = np.array([-(p2[1]-p1[1]), p2[0]-p1[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        
        # Project probe-p1 onto normal
        dist = np.dot(probe - p1, normal)
        
        # If we are inside, all distances should be same sign (depending on winding)
        # We just care about magnitude near boundary
        dist_prod *= (abs(dist) + 0.01)
        
    return 1.0 / dist_prod
