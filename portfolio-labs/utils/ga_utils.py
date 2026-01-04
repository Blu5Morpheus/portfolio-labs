import numpy as np

def lorenz_deriv(state, sigma, rho, beta):
    """
    Compute the derivatives for the Lorenz system.
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt, sigma, rho, beta):
    """
    Perform a single RK4 integration step.
    """
    k1 = lorenz_deriv(state, sigma, rho, beta)
    k2 = lorenz_deriv(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_deriv(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_deriv(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz_trajectory(x0, T, dt, sigma, rho, beta):
    """
    Generate the Lorenz trajectory.
    """
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    traj = np.zeros((steps, 3))
    
    curr = x0
    traj[0] = curr
    
    for i in range(1, steps):
        curr = rk4_step(curr, dt, sigma, rho, beta)
        traj[i] = curr
        
    return t, traj

def compute_tangents(trajectory):
    """
    Compute discrete tangent vectors v_t = p_{t+1} - p_t.
    """
    # Differences between consecutive points
    tangents = np.diff(trajectory, axis=0)
    return tangents

def compute_wedge_magnitude(tangents):
    """
    Compute magnitude of the wedge product between successive tangents.
    In 3D: ||v_t ^ v_{t+1}|| = ||v_t x v_{t+1}||
    """
    # We need v_t and v_{t+1}. 
    # v_t is tangents[:-1], v_{t+1} is tangents[1:]
    v_t = tangents[:-1]
    v_next = tangents[1:]
    
    # Compute cross product
    cross_products = np.cross(v_t, v_next)
    
    # Compute magnitudes
    wedge_mags = np.linalg.norm(cross_products, axis=1)
    
    return wedge_mags
