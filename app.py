import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="HMT Heat Solver", layout="wide")

st.title("2D Steady-State Heat Conduction: FDM vs Analytical")

# --- SIDEBAR: INPUT PARAMETERS ---
with st.sidebar:
    st.header("1. Boundary Conditions (°C)")
    t_top = st.number_input("Top Temp ($T_{top}$)", value=100.0)
    t_bottom = st.number_input("Bottom Temp ($T_{bottom}$)", value=20.0)
    t_side = st.number_input("Side Temps ($T_{left}=T_{right}$)", value=50.0)
    
    st.header("2. Grid Parameters")
    nx = st.slider("Grid Points X (nx)", 10, 60, 40)
    ny = st.slider("Grid Points Y (ny)", 10, 60, 40)
    
    st.header("3. Analytical Settings")
    n_terms = st.slider("Fourier Series Terms", 1, 100, 50)

# --- SOLVER FUNCTIONS ---

def solve_numerical(nx, ny, top, bottom, side):
    """Finite Difference Method (Gauss-Seidel)"""
    T = np.full((ny, nx), (top + bottom + side) / 3)
    T[-1, :] = top
    T[0, :] = bottom
    T[:, 0] = side
    T[:, -1] = side
    
    for _ in range(3000):
        T_old = T.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
        if np.max(np.abs(T - T_old)) < 1e-5:
            break
    return T

def solve_analytical(nx, ny, top, bottom, side, terms):
    """Fourier Series using Superposition relative to side temp"""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    T_ana = np.zeros((ny, nx))

    def fourier_contribution(X_g, Y_g, T_bc):
        res = np.zeros_like(X_g)
        for n in range(1, terms*2, 2):
            # Standard solution for 3 sides at 0, one side at T_bc
            term = (4 * T_bc) / (n * np.pi) * \
                   (np.sinh(n * np.pi * Y_g) / np.sinh(n * np.pi)) * \
                   np.sin(n * np.pi * X_g)
            res += term
        return res

    # Superposition: Top relative to side + Bottom relative to side + side baseline
    T_ana += fourier_contribution(X, Y, top - side)          # From Top
    T_ana += fourier_contribution(X, 1 - Y, bottom - side)    # From Bottom
    T_ana += side
    return T_ana

# --- CALCULATIONS ---
T_num = solve_numerical(nx, ny, t_top, t_bottom, t_side)
T_ana = solve_analytical(nx, ny, t_top, t_bottom, t_side, n_terms)
error = np.abs(T_num - T_ana)

# --- DISPLAY TABS ---
tab1, tab2, tab3 = st.tabs(["Numerical (FDM)", "Analytical", "Error Analysis"])

x_coords = np.linspace(0, 1, nx)
y_coords = np.linspace(0, 1, ny)

def create_heatmap(Z, title):
    fig = go.Figure(data=go.Heatmap(
        z=Z, x=x_coords, y=y_coords, colorscale='Viridis',
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Temp: %{z:.2f}°C<extra></extra>'
    ))
    fig.update_layout(title=title, width=600, height=500, xaxis_title="Width", yaxis_title="Height")
    return fig

with tab1:
    st.plotly_chart(create_heatmap(T_num, "FDM Solution"), use_container_width=True)
    st.info("Move your mouse/touch the map to see local temperatures.")

with tab2:
    st.plotly_chart(create_heatmap(T_ana, f"Analytical Solution ({n_terms} terms)"), use_container_width=True)

with tab3:
    st.plotly_chart(create_heatmap(error, "Absolute Error (|Num - Ana|)"), use_container_width=True)
    st.metric("Max Absolute Error", f"{np.max(error):.5f} °C")

st.divider()
st.markdown("### Engineering Project Summary")
cols = st.columns(3)
cols[0].write(f"**Grid Size:** {nx} x {ny}")
cols[1].write(f"**Method:** Gauss-Seidel")
cols[2].write(f"**Convergence:** 1e-5 tolerance")
