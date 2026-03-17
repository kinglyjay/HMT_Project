import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="HMT: FDM vs SOR vs Analytical", layout="wide")

st.title("2D Heat Conduction: Comparison of Solvers")
st.markdown("Comparing **Gauss-Seidel**, **SOR (Successive Over-Relaxation)**, and the **Analytical Fourier Series**.")

#sidebar controls
with st.sidebar:
    st.header("1. Boundary Conditions (°C)")
    t_top = st.number_input("Top Temp", value=100.0)
    t_bottom = st.number_input("Bottom Temp", value=20.0)
    t_side = st.number_input("Side Temps (L/R)", value=50.0)
    
    st.header("2. Numerical Controls")
    nx = st.slider("Grid Points (nx=ny)", 10, 60, 40)
    solver_type = st.radio("Select Solver", ["Gauss-Seidel", "SOR"])
    omega = st.slider("Relaxation Factor (ω)", 1.0, 1.95, 1.5, step=0.05) if solver_type == "SOR" else 1.0
    
    st.header("3. Analytical Terms")
    n_terms = st.slider("Fourier Terms", 1, 100, 50)

#solverpart

def solve_numerical(nx, ny, top, bottom, side, method="Gauss-Seidel", w=1.0):
    T = np.full((ny, nx), (top + bottom + side) / 3)
    T[-1, :] = top
    T[0, :] = bottom
    T[:, 0] = side
    T[:, -1] = side
    
    tol = 1e-5
    max_iter = 5000
    it_count = 0
    start_time = time.time()

    for it in range(max_iter):
        T_old = T.copy()
        it_count = it
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # The GS base calculation
                T_gs = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
                
                if method == "SOR":
                    # SOR Formula: T_new = (1-w)*T_old + w*T_gs
                    T[i, j] = (1 - w) * T[i, j] + w * T_gs
                else:
                    T[i, j] = T_gs
        
        if np.max(np.abs(T - T_old)) < tol:
            break
            
    return T, it_count, time.time() - start_time

def solve_analytical(nx, ny, top, bottom, side, terms):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    T_ana = np.full((ny, nx), side)

    def fourier_sum(X_g, Y_g, T_bc):
        res = np.zeros_like(X_g)
        for n in range(1, terms*2, 2):
            term = (4 * T_bc) / (n * np.pi) * \
                   (np.sinh(n * np.pi * Y_g) / np.sinh(n * np.pi)) * \
                   np.sin(n * np.pi * X_g)
            res += term
        return res

    T_ana += fourier_sum(X, Y, top - side)
    T_ana += fourier_sum(X, 1 - Y, bottom - side)
    T_ana += side
    return T_ana

#body
T_num, iterations, exec_time = solve_numerical(nx, nx, t_top, t_bottom, t_side, solver_type, omega)
T_ana = solve_analytical(nx, nx, t_top, t_bottom, t_side, n_terms)
error = np.abs(T_num - T_ana)

#graphsandplots
col1, col2 = st.columns(2)

with col1:
    fig_num = go.Figure(data=go.Heatmap(z=T_num, colorscale='Inferno'))
    fig_num.update_layout(title=f"Numerical: {solver_type}", width=500, height=500)
    st.plotly_chart(fig_num)
    
    st.metric("Iterations", iterations)
    st.metric("Execution Time", f"{exec_time:.4f}s")

with col2:
    fig_ana = go.Figure(data=go.Heatmap(z=T_ana, colorscale='Inferno'))
    fig_ana.update_layout(title="Analytical Solution", width=500, height=500)
    st.plotly_chart(fig_ana)
    
    st.metric("Max Error", f"{np.max(error):.5e} °C")

st.divider()
st.subheader("Performance Insights")
st.write(f"Using **{solver_type}**, the solver reached steady state in **{iterations}** steps. "
         f"Notice how increasing **ω** (between 1.5 and 1.9) significantly drops the iteration count compared to Gauss-Seidel (ω=1).")
