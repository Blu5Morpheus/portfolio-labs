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
