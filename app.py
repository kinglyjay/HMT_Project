import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- 1. The Solver Logic (Fixed Omega) ---
def solve_heat_sor(nx, ny, top, bottom, left, right, tol=1e-4, max_iter=5000):
    # Omega is now hardcoded to 1.8 for performance
    omega = 1.8 
    T = np.zeros((nx, ny))
    T[-1, :] = top
    T[0, :] = bottom
    T[:, 0] = left
    T[:, -1] = right

    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                gs_term = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
                new_val = (1 - omega) * T[i, j] + omega * gs_term
                diff = abs(new_val - T[i, j])
                if diff > max_diff: max_diff = diff
                T[i, j] = new_val
        if max_diff < tol:
            break
    return T, it

# --- 2. Custom CSS for Black/White Theme ---
st.set_page_config(page_title="IITP Heat Solver", layout="wide")

st.markdown("""
    <style>
    /* Main background to Black */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    /* Sidebar background to Black */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #FFFFFF;
    }
    /* Make all text White */
    h1, h2, h3, p, span, label {
        color: #FFFFFF !important;
    }
    /* Style the sliders/inputs to stand out */
    .stSlider, .stNumberInput {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. UI Layout ---
st.title("2D Heat Equation Solver")
st.markdown("Steady-state thermal analysis. Adjust boundary conditions in the sidebar.")

# Sidebar Inputs
st.sidebar.header("Parameters")
nx = st.sidebar.slider("Nodes in X", 10, 80, 40)
ny = st.sidebar.slider("Nodes in Y", 10, 80, 40)

t_top = st.sidebar.number_input("Top Temp (°C)", value=100.0)
t_bottom = st.sidebar.number_input("Bottom Temp (°C)", value=0.0)
t_left = st.sidebar.number_input("Left Temp (°C)", value=75.0)
t_right = st.sidebar.number_input("Right Temp (°C)", value=50.0)

if st.button("Solve & Generate Heatmap"):
    with st.spinner('Calculating...'):
        final_T, iterations = solve_heat_sor(nx, ny, t_top, t_bottom, t_left, t_right)
    
    st.success(f"Converged in {iterations} iterations.")

    # --- 4. Interactive Color Heatmap ---
    # Using 'Jet' or 'Viridis' for color while the rest of the site is B&W
    fig = go.Figure(data=go.Heatmap(
        z=final_T,
        colorscale='Jet', 
        hovertemplate='Node X: %{x}<br>Node Y: %{y}<br>Temp: %{z:.2f}°C<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)