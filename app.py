import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="2D Convective Heat Solver", layout="wide")

st.title("2D Heat Conduction: Robin Boundary Condition")
st.markdown("Comparing **Numerical (FDM)** and **Analytical (Fourier Series)** solutions.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Boundary Temperatures (°C)")
    t_inf = st.number_input("Ambient Fluid Temp (T_inf)", value=100.0)
    t_bottom = st.number_input("Bottom Base Temp", value=20.0)
    t_side = st.number_input("Left/Right Side Temp", value=50.0)
    
    st.header("2. Material & Convection")
    h_coeff = st.number_input("Convection Coeff (h)", value=50.0)
    k_cond = st.number_input("Thermal Cond. (k)", value=15.0)
    
    st.header("3. Numerical Controls")
    nx = st.slider("Grid Resolution (N x N)", 10, 60, 40)
    solver_type = st.radio("Select Solver", ["Gauss-Seidel", "SOR"])
    w_opt = 2 / (1 + np.sin(np.pi/nx))
    omega = st.slider("Relaxation Factor (ω)", 1.0, 1.95, float(round(w_opt, 2))) if solver_type == "SOR" else 1.0
    
    st.header("4. Analytical Settings")
    n_terms = st.slider("Fourier Terms", 10, 100, 50)

# --- NUMERICAL SOLVER ---
def solve_numerical_convective(n, t_inf, t_b, t_s, h, k, method, w):
    dx = 1.0 / (n - 1)
    Bi = (h * dx) / k
    T = np.full((n, n), (t_inf + t_b + t_s) / 3.0)
    T[0, :] = t_b
    T[:, 0] = t_s
    T[:, -1] = t_s
    
    tol = 1e-5
    max_iter = 5000
    start_time = time.time()

    for it in range(max_iter):
        T_old = T.copy()
        # Internal update
        T_int = 0.25 * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2])
        if method == "SOR":
            T[1:-1, 1:-1] = (1 - w) * T[1:-1, 1:-1] + w * T_int
        else:
            T[1:-1, 1:-1] = T_int
            
        # Top Convection update
        T_top_new = (2*T[-2, 1:-1] + T[-1, :-2] + T[-1, 2:] + 2*Bi*t_inf) / (4 + 2*Bi)
        if method == "SOR":
            T[-1, 1:-1] = (1 - w) * T[-1, 1:-1] + w * T_top_new
        else:
            T[-1, 1:-1] = T_top_new

        if np.max(np.abs(T - T_old)) < tol:
            return T, it, time.time() - start_time
    return T, max_iter, time.time() - start_time

# --- STABILIZED ANALYTICAL SOLVER ---
def solve_analytical_convective(n, t_inf, t_b, t_s, h, k, terms):
    L, H = 1.0, 1.0
    x = np.linspace(0, L, n)
    y = np.linspace(0, H, n)
    X, Y = np.meshgrid(x, y)
    
    th_b, th_inf = t_b - t_s, t_inf - t_s
    C = h / k
    Theta = np.zeros_like(X)
    
    for i in range(1, terms + 1):
        m = 2 * i - 1
        lam = m * np.pi / L
        Bn = (4 * th_b) / (m * np.pi)
        Cn_inf = (4 * C * th_inf) / (m * np.pi)
        
        # Stability: calculate sinh/cosh relative to exp(lam*H)
        sinh_H = (1 - np.exp(-2*lam*H)) / (1 + np.exp(-2*lam*H))
        
        num = (Cn_inf / np.cosh(lam*H)) - Bn * (lam * sinh_H + C)
        den = lam + C * sinh_H
        An = num / den
        
        # Using exponential forms that don't explode (Y-H is always <= 0)
        term = (An * (np.exp(lam*(Y-H)) - np.exp(-lam*(Y+H)))/2 + 
                Bn * (np.exp(lam*(Y-H)) + np.exp(-lam*(Y+H)))/2) * np.cosh(lam*H) * np.sin(lam*X)
        Theta += term
        
    return Theta + t_s

# --- EXECUTION ---
T_num, it_num, time_num = solve_numerical_convective(nx, t_inf, t_bottom, t_side, h_coeff, k_cond, solver_type, omega)
T_ana = solve_analytical_convective(nx, t_inf, t_bottom, t_side, h_coeff, k_cond, n_terms)

# --- VISUALIZATION ---
col1, col2 = st.columns(2)

# FIXED PLOTLY STYLE DICTIONARY
# Nested properties like showlines must go inside 'contours'
style = dict(
    ncontours=20,
    colorscale='Inferno',
    contours=dict(showlines=True),
    line=dict(width=0.5, color='white'),
    contours_coloring='heatmap'
)

with col1:
    st.subheader("Numerical Solution")
    fig1 = go.Figure(data=go.Contour(z=T_num, **style))
    fig1.update_layout(xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig1, use_container_width=True)
    st.write(f"Steps: {it_num} | Time: {time_num:.4f}s")

with col2:
    st.subheader("Analytical Solution")
    fig2 = go.Figure(data=go.Contour(z=T_ana, **style))
    fig2.update_layout(xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig2, use_container_width=True)
    st.write(f"Max Temp: {np.max(T_ana):.2f}°C")
