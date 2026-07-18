import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import VarQITE
from qiskit_algorithms.time_evolvers import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

# ============================================================
# GLOBAL CONFIGURATION (EASY TO MODIFY)
# ============================================================

VERBOSE = False  # set True for detailed debug prints

# ----- Multi-Agent System -----
N_AGENTS = 5

# Ring topology adjacency matrix
A = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
], dtype=float)

D = np.diag(A.sum(axis=1))
L = D - A

# ----- Physical drag coefficients (fixed, "real" plant parameters) -----
A_DRAG_LIN = 0.5    # linear drag coefficient
B_DRAG_QUAD = 0.05  # quadratic drag coefficient

# ----- Initial condition for the MAS (positions + velocities) -----
# z = [x; v]
X0_POS = np.array([5.0, -4.0, 3.0, -2.0, 1.0])
X0_VEL = np.array([0.0, 1.5, -1.0, 0.5, -0.5])
X0_GLOBAL = np.concatenate([X0_POS, X0_VEL])

# ----- Lyapunov-aware cost (both for BH and Stage 2) -----
T_HORIZON_COST = 0.25
N_TIME_COST = 150
W_Z = 1.0
W_U = 0.1
W_LYAP = 1.0

# ----- Global parameter ranges (Stage 1 initial box) -----
# Parameters: [Kp, Kd, theta_x2, theta_v2, theta_x4]
PARAM_NAMES = ["Kp", "Kd", "theta_x2", "theta_v2", "theta_x4"]

P_MIN_GLOBAL = np.array([0.0, 0.0,   0.0,   0.0,   0.0])
P_MAX_GLOBAL = np.array([50.0, 50.0, 50.0, 40.0, 20.0])

# ----- Stage 1: Black Hole (BH) metaheuristic -----
BH_POP_SIZE = 20
BH_MAX_ITERS = 200
BH_FREEZE_WIDTH = 5.0

# ----- Stage 2: Dynamic quantum Lyapunov synthesis -----
T_TOTAL = 20.0
DT_DECISION = 0.5
QITE_TAU = 3.0
QITE_STEPS = 60
QITE_REPS = 2
QITE_SEED_BASE = 42

# Stopping criterion for consensus (positions + velocities)
# error = sqrt( ||Lx||_2^2 + ||Lv||_2^2 ); stop when <= CONS_TOL
CONS_TOL = 1e-4

# Hamiltonian fitting / candidate evaluation
TRAIN_SAMPLE_FACTOR = 4
MIN_TRAIN_SAMPLES = 64
TOP_K_CANDIDATES = 32

# ----- Bit allocation rules -----
BIT_WIDTH_THRESHOLDS = [5.0, 20.0]
BIT_ALLOCATION = [2, 3, 4]
MAX_BITS_PER_PARAM = 4


# ============================================================
# IEEE/TCST PUBLICATION PLOTTING SETTINGS
# ============================================================

def configure_ieee_tcst_plot_style():
    """
    IEEE/TCST-oriented plotting style with larger fonts and thicker curves
    for readability when figures are used as small subfigures.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # --- Larger text for small subfigures ---
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,

        # Thicker curves, markers, and axes
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.4,
        "lines.markersize": 6.5,

        # Tick style
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        # High-quality vector PDF export
        "savefig.dpi": 1200,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Legend style
        "legend.frameon": True,
        "legend.framealpha": 0.98,
        "legend.edgecolor": "0.25",
    })

IEEE_COLORS = {
    "blue":   "#0072B2",
    "orange": "#D55E00",
    "green":  "#009E73",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#E69F00",
    "black":  "#000000",
    "gray":   "#666666",
}

IEEE_AGENT_COLORS = [
    IEEE_COLORS["blue"],
    IEEE_COLORS["orange"],
    IEEE_COLORS["green"],
    IEEE_COLORS["purple"],
    IEEE_COLORS["sky"],
]

IEEE_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def format_ieee_axes(ax, xlabel=None, ylabel=None):
    """Apply consistent IEEE-style axis formatting."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=3)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=3)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.85)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.45)
    ax.minorticks_on()

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=4.5,
        width=1.0,
        pad=2,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=0.8,
        pad=2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def add_ieee_legend(ax, loc="best", ncol=1):
    """Add compact publication-quality legend with larger readable text."""
    leg = ax.legend(
        loc=loc,
        ncol=ncol,
        handlelength=2.4,
        borderpad=0.4,
        columnspacing=0.8,
        labelspacing=0.3,
        handletextpad=0.5,
        frameon=True,
        framealpha=0.98,
        edgecolor="0.25",
        fancybox=False,
    )

    if leg is not None:
        leg.get_frame().set_linewidth(0.8)

    return leg

def save_ieee_figure(fig, filename_base):
    """
    Save a figure as a publication-quality vector PDF.

    PDF is preferred for IEEE papers because text, axes, and curves remain sharp.
    """
    fig.savefig(
        f"{filename_base}.pdf",
        format="pdf",
        dpi=1200,
        bbox_inches="tight",
        pad_inches=0.02,
    )


# ============================================================
# 1) Closed-loop dynamics and Lyapunov functions
# ============================================================

def closed_loop_dynamics(t, z, Kp, Kd):
    """
    Second-order MAS with linear + quadratic drag and distributed consensus control.

    For each agent i:

        x_i' = v_i
        v_i' = -A_DRAG_LIN * v_i - B_DRAG_QUAD * |v_i| v_i + u_i

    where the control is

        u_i = -Kp * sum_{j in N_i} (x_i - x_j)
              -Kd * sum_{j in N_i} (v_i - v_j)

    vector form:
        u = -Kp * (L x) - Kd * (L v)
    """
    z = np.asarray(z)
    x = z[:N_AGENTS]
    v = z[N_AGENTS:]

    z_x = L @ x
    z_v = L @ v

    u = -Kp * z_x - Kd * z_v

    dxdt = v
    dvdt = -A_DRAG_LIN * v - B_DRAG_QUAD * np.abs(v) * v + u

    return np.concatenate([dxdt, dvdt])


def control_law(z, Kp, Kd):
    """
    Compute control u(t) for a given state z and gains (Kp, Kd).
    z can be shape (2N,) or (2N, T).
    """
    z = np.asarray(z)

    if z.ndim == 1:
        x = z[:N_AGENTS]
        v = z[N_AGENTS:]

        z_x = L @ x
        z_v = L @ v

        u = -Kp * z_x - Kd * z_v
        return u

    else:
        x = z[:N_AGENTS, :]
        v = z[N_AGENTS:, :]

        z_x = L @ x
        z_v = L @ v

        u = -Kp * z_x - Kd * z_v
        return u


def lyapunov_V(z, theta_x2, theta_v2, theta_x4):
    """
    Lyapunov candidate in disagreement coordinates:

        z_x = L x,  z_v = L v

        V(z) = (theta_x2/2) * ||z_x||^2
             + (theta_v2/2) * ||z_v||^2
             + (theta_x4/4) * sum_i z_x_i^4
    """
    z = np.asarray(z)
    x = z[:N_AGENTS]
    v = z[N_AGENTS:]

    z_x = L @ x
    z_v = L @ v

    term_quad_x = 0.5 * theta_x2 * np.sum(z_x**2)
    term_quad_v = 0.5 * theta_v2 * np.sum(z_v**2)
    term_quartic_x = 0.25 * theta_x4 * np.sum(z_x**4)

    return term_quad_x + term_quad_v + term_quartic_x


def lyapunov_dVdt(z, dzdt, theta_x2, theta_v2, theta_x4):
    """
    Time derivative of V along trajectories.
    """
    z = np.asarray(z)
    dzdt = np.asarray(dzdt)

    x = z[:N_AGENTS]
    v = z[N_AGENTS:]
    dxdt = dzdt[:N_AGENTS]
    dvdt = dzdt[N_AGENTS:]

    z_x = L @ x
    z_v = L @ v

    grad_x = L.T @ (theta_x2 * z_x + theta_x4 * z_x**3)
    grad_v = L.T @ (theta_v2 * z_v)

    dVdt_val = float(np.dot(grad_x, dxdt) + np.dot(grad_v, dvdt))
    return dVdt_val


def simulate_horizon_cost(z0,
                          Kp, Kd, theta_x2, theta_v2, theta_x4,
                          t_horizon=T_HORIZON_COST,
                          n_time=N_TIME_COST,
                          w_z=W_Z,
                          w_u=W_U,
                          w_lyap=W_LYAP):
    """
    Short-horizon Lyapunov-aware cost from z0 = [x; v].
    """
    t_span = (0.0, t_horizon)
    t_eval = np.linspace(t_span[0], t_span[1], n_time)

    def dyn(t, z):
        return closed_loop_dynamics(t, z, Kp, Kd)

    sol = solve_ivp(
        dyn,
        t_span,
        z0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        return 1e6

    t = sol.t
    Ztraj = sol.y  # shape (2N_AGENTS, len(t))

    if np.any(np.abs(Ztraj) > 1e6):
        return 1e6

    Xtraj = Ztraj[:N_AGENTS, :]
    Vtraj = Ztraj[N_AGENTS:, :]

    Zx = L @ Xtraj
    Zv = L @ Vtraj
    U = control_law(Ztraj, Kp, Kd)

    z_x_norm_sq = np.sum(Zx**2, axis=0)
    z_v_norm_sq = np.sum(Zv**2, axis=0)
    u_norm_sq = np.sum(U**2, axis=0)

    integrand_perf = w_z * (z_x_norm_sq + z_v_norm_sq) + w_u * u_norm_sq

    dVdt_vals = []
    for idx, tt in enumerate(t):
        z_t = Ztraj[:, idx]
        dzdt_t = dyn(tt, z_t)
        dVdt_vals.append(lyapunov_dVdt(z_t, dzdt_t, theta_x2, theta_v2, theta_x4))

    dVdt_vals = np.array(dVdt_vals)

    lyap_violation = np.maximum(0.0, dVdt_vals)
    integrand_lyap = lyap_violation**2

    J_perf = float(np.trapz(integrand_perf, t))
    J_lyap = float(np.trapz(integrand_lyap, t))
    J_total = J_perf + w_lyap * J_lyap

    return J_total


# ============================================================
# 2) Stage 1: Black Hole (BH) range calibration
# ============================================================

def bh_initialize_population(p_min_init, p_max_init):
    """
    Initialize BH star population uniformly in [p_min_init, p_max_init].
    """
    n_params = len(PARAM_NAMES)
    stars = np.random.uniform(
        low=p_min_init,
        high=p_max_init,
        size=(BH_POP_SIZE, n_params)
    )
    return stars


def bh_evaluate_population(stars, z0):
    """
    Evaluate black-box cost for each star.
    """
    costs = np.zeros(len(stars))

    for i, p in enumerate(stars):
        Kp, Kd, theta_x2, theta_v2, theta_x4 = p
        costs[i] = simulate_horizon_cost(z0, Kp, Kd, theta_x2, theta_v2, theta_x4)

    return costs


def bh_range_calibration(p_min_init, p_max_init, z0):
    """
    Simplified Black Hole metaheuristic to shrink parameter ranges
    around the current state z0.
    """
    np.random.seed(0)

    n_params = len(PARAM_NAMES)
    stars = bh_initialize_population(p_min_init, p_max_init)
    active = np.ones(n_params, dtype=bool)
    p_min = p_min_init.copy()
    p_max = p_max_init.copy()

    for it in range(BH_MAX_ITERS):
        costs = bh_evaluate_population(stars, z0)
        best_idx = np.argmin(costs)
        black_hole = stars[best_idx].copy()
        best_cost = costs[best_idx]

        if VERBOSE:
            print(f"[BH] Iter {it+1}/{BH_MAX_ITERS}, best_cost={best_cost:.4e}")

        for i in range(BH_POP_SIZE):
            if i == best_idx:
                continue

            rand_vec = np.random.rand(np.sum(active))
            stars[i, active] = (
                stars[i, active]
                + rand_vec * (black_hole[active] - stars[i, active])
            )

        for j in range(n_params):
            stars[:, j] = np.clip(stars[:, j], p_min[j], p_max[j])

        for j in range(n_params):
            if not active[j]:
                continue

            cur_min = np.min(stars[:, j])
            cur_max = np.max(stars[:, j])

            p_min[j] = max(cur_min, P_MIN_GLOBAL[j])
            p_max[j] = min(cur_max, P_MAX_GLOBAL[j])
            width = p_max[j] - p_min[j]

            if width <= BH_FREEZE_WIDTH:
                active[j] = False
                if VERBOSE:
                    print(
                        f"[BH] Param {PARAM_NAMES[j]} frozen with range "
                        f"[{p_min[j]:.3f}, {p_max[j]:.3f}]"
                    )

        if not np.any(active):
            if VERBOSE:
                print("[BH] All parameters frozen, stopping BH for this step.")
            break

    if VERBOSE:
        print("[BH] Final ranges for this step:")
        for name, mn, mx in zip(PARAM_NAMES, p_min, p_max):
            print(f"   {name}: [{mn:.3f}, {mx:.3f}] width={mx-mn:.3f}")

    return p_min, p_max


# ============================================================
# 3) Bit allocation & parameter encoding
# ============================================================

def choose_bits_for_width(width):
    """
    Choose the number of bits based on interval width.
    """
    if width <= BIT_WIDTH_THRESHOLDS[0]:
        return min(BIT_ALLOCATION[0], MAX_BITS_PER_PARAM)
    elif width <= BIT_WIDTH_THRESHOLDS[1]:
        return min(BIT_ALLOCATION[1], MAX_BITS_PER_PARAM)
    else:
        return min(BIT_ALLOCATION[2], MAX_BITS_PER_PARAM)


def allocate_bits_for_parameters(p_min, p_max):
    """
    For each parameter, compute width and choose bits.
    """
    widths = p_max - p_min
    bits_per_param = []

    for w in widths:
        bits_per_param.append(choose_bits_for_width(w))

    num_qubits = int(np.sum(bits_per_param))

    if VERBOSE:
        print("[Bits] Allocation per param (this step):")
        for name, w, b in zip(PARAM_NAMES, widths, bits_per_param):
            print(f"   {name}: width={w:.3f}, bits={b}")
        print(f"   Total qubits = {num_qubits}")

    return bits_per_param, num_qubits


def decode_bitstring_to_params(bitstr, p_min, p_max, bits_per_param):
    """
    Decode a bitstring to continuous parameters.
    Returns [Kp, Kd, theta_x2, theta_v2, theta_x4].
    """
    assert len(bitstr) == int(np.sum(bits_per_param))

    p_min = np.asarray(p_min)
    p_max = np.asarray(p_max)
    params = []

    idx = 0
    for i, n_bits in enumerate(bits_per_param):
        bits_i = bitstr[idx: idx + n_bits]
        idx += n_bits

        v = int(bits_i, 2)
        levels = 2**n_bits - 1

        if levels <= 0:
            p_val = p_min[i]
        else:
            p_val = p_min[i] + (p_max[i] - p_min[i]) * (v / levels)

        params.append(p_val)

    return params


# ============================================================
# 4) Diagonal Pauli basis & Hamiltonian fit
# ============================================================

def build_diagonal_pauli_basis(num_qubits):
    """
    Diagonal Pauli basis: {I...I, all single Z_i, all pairwise Z_i Z_j}.
    """
    n = num_qubits
    paulis = []

    paulis.append("I" * n)

    for i in range(n):
        s = ["I"] * n
        s[i] = "Z"
        paulis.append("".join(s))

    for i in range(n):
        for j in range(i + 1, n):
            s = ["I"] * n
            s[i] = "Z"
            s[j] = "Z"
            paulis.append("".join(s))

    return paulis


def eigenvalue_of_pauli_on_bitstring(pauli_label, bitstr):
    """
    Eigenvalue of a diagonal Pauli (I/Z only) on |bitstr>.
    """
    n = len(bitstr)
    assert len(pauli_label) == n

    bits = [int(b) for b in bitstr]
    eig = 1.0

    # Qiskit convention: rightmost bit = qubit 0
    for j in range(n):
        p = pauli_label[n - 1 - j]
        if p == "Z":
            eig *= 1.0 if bits[-1 - j] == 0 else -1.0
        elif p == "I":
            continue
        else:
            raise ValueError("Non-diagonal Pauli encountered.")

    return eig


def sample_bitstrings(num_qubits, num_samples):
    """
    Uniformly sample unique bitstrings of length num_qubits.
    """
    max_possible = 2**num_qubits
    num_samples = min(num_samples, max_possible)

    bitstrings = set()
    while len(bitstrings) < num_samples:
        s = "".join(np.random.choice(["0", "1"], size=num_qubits))
        bitstrings.add(s)

    return list(bitstrings)


def fit_diagonal_hamiltonian_sampled(z0, p_min, p_max, bits_per_param,
                                     paulis, num_qubits):
    """
    Fit H = sum_k h_k P_k using a sampled subset of bitstrings.
    """
    n_basis = len(paulis)
    num_samples = max(TRAIN_SAMPLE_FACTOR * n_basis, MIN_TRAIN_SAMPLES)

    if VERBOSE:
        print(
            f"[H-fit] num_qubits={num_qubits}, "
            f"n_basis={n_basis}, num_samples={num_samples}"
        )

    sampled_bitstrings = sample_bitstrings(num_qubits, num_samples)
    num_samples = len(sampled_bitstrings)

    M = np.zeros((num_samples, n_basis), dtype=float)
    J_vec = np.zeros(num_samples, dtype=float)

    for idx, s in enumerate(sampled_bitstrings):
        Kp, Kd, theta_x2, theta_v2, theta_x4 = decode_bitstring_to_params(
            s, p_min, p_max, bits_per_param
        )

        J = simulate_horizon_cost(z0, Kp, Kd, theta_x2, theta_v2, theta_x4)
        J_vec[idx] = J

        for k, P in enumerate(paulis):
            M[idx, k] = eigenvalue_of_pauli_on_bitstring(P, s)

    h, *_ = np.linalg.lstsq(M, J_vec, rcond=None)

    if VERBOSE:
        print("[H-fit] First few coefficients:")
        for coef, P in list(zip(h, paulis))[:10]:
            print(f"   {P}: {coef:.6f}")

    H = SparsePauliOp.from_list(list(zip(paulis, h)))
    return H


# ============================================================
# 5) QITE-based search for best bitstring
# ============================================================

def qite_select_best_bitstring(H, z0, p_min, p_max, bits_per_param,
                               tau=QITE_TAU,
                               steps=QITE_STEPS,
                               reps=QITE_REPS,
                               seed=0):
    """
    Run VarQITE and select the best bitstring according to the true cost.
    """
    np.random.seed(seed)
    num_qubits = int(np.sum(bits_per_param))

    ansatz = EfficientSU2(num_qubits, reps=reps, entanglement="linear")
    n_params = ansatz.num_parameters

    if VERBOSE:
        print(f"[QITE] num_qubits={num_qubits}, ansatz params={n_params}")

    init_params = 0.02 * np.random.randn(n_params)

    est = Estimator()
    principle = ImaginaryMcLachlanPrinciple()

    varqite = VarQITE(
        ansatz=ansatz,
        initial_parameters=init_params,
        variational_principle=principle,
        estimator=est,
        num_timesteps=steps,
    )

    problem = TimeEvolutionProblem(hamiltonian=H, time=tau)
    result = varqite.evolve(problem)

    final_circ = result.evolved_state
    sv = Statevector.from_instruction(final_circ)
    probs = sv.probabilities_dict()

    sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    candidates = sorted_items[:min(TOP_K_CANDIDATES, len(sorted_items))]

    best_s, best_J = None, float("inf")

    if VERBOSE:
        print("[QITE] Evaluating true cost on candidates:")

    for s, p_prob in candidates:
        Kp, Kd, theta_x2, theta_v2, theta_x4 = decode_bitstring_to_params(
            s, p_min, p_max, bits_per_param
        )

        J = simulate_horizon_cost(z0, Kp, Kd, theta_x2, theta_v2, theta_x4)

        if VERBOSE:
            print(f"   s={s}, p_prob={p_prob:.4f}, J={J:.4f}")

        if J < best_J:
            best_s, best_J = s, J

    energy = est.run([final_circ], [H]).result().values[0]

    if VERBOSE:
        print(f"[QITE] Best bitstring {best_s} with J={best_J:.4f}, <H>={energy:.4f}")

    return best_s, best_J


# ============================================================
# 6) Full loop: BH at every step + dynamic QITE + early stopping
# ============================================================

def dynamic_quantum_lyapunov_synthesis_with_repeated_BH():
    """
    Full algorithm for second-order MAS with drag.

    Additional stored histories for plotting:
      - Kp_hist, Kd_hist
      - theta_x2_hist, theta_v2_hist, theta_x4_hist
      - cost_hist
      - consensus error history
      - redesign_times
    """
    print("=== Dynamic Quantum Lyapunov Synthesis with BH at Every Step ===")
    print("=== Second-order MAS with linear + quadratic drag ===")

    p_min_curr = P_MIN_GLOBAL.copy()
    p_max_curr = P_MAX_GLOBAL.copy()

    z_current = X0_GLOBAL.copy()

    t_global = [0.0]
    Z_global = [z_current.copy()]
    U_global = [control_law(z_current, 0.0, 0.0)]  # dummy first control

    # Histories for requested plots
    Kp_hist = []
    Kd_hist = []
    theta_x2_hist = []
    theta_v2_hist = []
    theta_x4_hist = []
    cost_hist = []
    cons_err_hist = []
    redesign_times = []

    max_steps = int(T_TOTAL / DT_DECISION)

    for k_step in range(max_steps):
        print(f"\n=== Decision step {k_step+1}/{max_steps} ===")

        if VERBOSE:
            print(f"   Current state z_k = {z_current}")

        # ---------------------------------------------------
        # Stage 1: BH range calibration
        # ---------------------------------------------------
        p_min_step, p_max_step = bh_range_calibration(
            p_min_curr,
            p_max_curr,
            z_current
        )
        p_min_curr = p_min_step
        p_max_curr = p_max_step

        print("   Updated parameter ranges after BH:")
        for name, mn, mx in zip(PARAM_NAMES, p_min_curr, p_max_curr):
            print(f"      {name}: [{mn:.3f}, {mx:.3f}] (width={mx-mn:.3f})")

        bits_per_param, num_qubits = allocate_bits_for_parameters(
            p_min_curr,
            p_max_curr
        )
        paulis = build_diagonal_pauli_basis(num_qubits)

        # ---------------------------------------------------
        # Stage 2: QITE-based synthesis
        # ---------------------------------------------------
        H = fit_diagonal_hamiltonian_sampled(
            z_current,
            p_min_curr,
            p_max_curr,
            bits_per_param,
            paulis,
            num_qubits
        )

        best_s, best_J = qite_select_best_bitstring(
            H,
            z_current,
            p_min_curr,
            p_max_curr,
            bits_per_param,
            tau=QITE_TAU,
            steps=QITE_STEPS,
            reps=QITE_REPS,
            seed=QITE_SEED_BASE + k_step
        )

        Kp, Kd, theta_x2, theta_v2, theta_x4 = decode_bitstring_to_params(
            best_s,
            p_min_curr,
            p_max_curr,
            bits_per_param
        )

        print(
            f"   Selected params: s={best_s}, "
            f"Kp={Kp:.3f}, Kd={Kd:.3f}, "
            f"theta_x2={theta_x2:.3f}, theta_v2={theta_v2:.3f}, "
            f"theta_x4={theta_x4:.3f}, J={best_J:.4f}"
        )

        # Save redesign histories
        Kp_hist.append(Kp)
        Kd_hist.append(Kd)
        theta_x2_hist.append(theta_x2)
        theta_v2_hist.append(theta_v2)
        theta_x4_hist.append(theta_x4)
        cost_hist.append(best_J)

        # ---------------------------------------------------
        # Integrate true closed-loop system over one decision interval
        # ---------------------------------------------------
        t_start = t_global[-1]
        t_end = t_start + DT_DECISION
        redesign_times.append(t_end)

        t_span = (t_start, t_end)
        t_eval = np.linspace(t_start, t_end, 200)

        def dyn_interval(t, z):
            return closed_loop_dynamics(t, z, Kp, Kd)

        sol = solve_ivp(
            dyn_interval,
            t_span,
            z_current,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )

        if not sol.success:
            print("   Integration failed in interval:", sol.message)
            break

        t_local = sol.t
        Z_local = sol.y
        U_local = control_law(Z_local, Kp, Kd)

        if k_step == 0 and len(t_global) == 1:
            t_global = list(t_local)
            Z_global = [Z_local[:, i] for i in range(Z_local.shape[1])]
            U_global = [U_local[:, i] for i in range(U_local.shape[1])]
        else:
            t_global.extend(list(t_local[1:]))
            Z_global.extend([Z_local[:, i] for i in range(1, Z_local.shape[1])])
            U_global.extend([U_local[:, i] for i in range(1, U_local.shape[1])])

        z_current = Z_local[:, -1].copy()

        x_end = z_current[:N_AGENTS]
        v_end = z_current[N_AGENTS:]

        cons_err = np.sqrt(
            np.linalg.norm(L @ x_end, 2)**2
            + np.linalg.norm(L @ v_end, 2)**2
        )

        cons_err_hist.append(cons_err)

        print(f"   State at end of interval: x={x_end}, v={v_end}")
        print(f"   Consensus error sqrt(||Lx||^2 + ||Lv||^2) = {cons_err:.3e}")

        if cons_err <= CONS_TOL:
            print("   Consensus error below tolerance; stopping simulation.")
            break

    t_global = np.array(t_global)
    Z_global = np.array(Z_global).T
    U_global = np.array(U_global).T

    Kp_hist = np.array(Kp_hist)
    Kd_hist = np.array(Kd_hist)
    theta_x2_hist = np.array(theta_x2_hist)
    theta_v2_hist = np.array(theta_v2_hist)
    theta_x4_hist = np.array(theta_x4_hist)
    cost_hist = np.array(cost_hist)
    cons_err_hist = np.array(cons_err_hist)
    redesign_times = np.array(redesign_times)

    return (
        t_global,
        Z_global,
        U_global,
        Kp_hist,
        Kd_hist,
        theta_x2_hist,
        theta_v2_hist,
        theta_x4_hist,
        cost_hist,
        cons_err_hist,
        redesign_times,
        p_min_curr,
        p_max_curr
    )


# ============================================================
# 7) Plot and save all requested TCST-style figures
# ============================================================

def plot_and_save_all_results(t, Z, U,
                              Kp_hist, Kd_hist,
                              theta_x2_hist, theta_v2_hist, theta_x4_hist,
                              cost_hist, cons_err_hist,
                              redesign_times):
    """
    Save these TCST-style plots:
      1. Position trajectories
      2. Velocity trajectories
      3. Control signals
      4. Consensus error history
      5. Controller-parameter redesign
      6. Lyapunov-parameter redesign
      7. Short-horizon cost history
    """
    configure_ieee_tcst_plot_style()

    X = Z[:N_AGENTS, :]
    V = Z[N_AGENTS:, :]

    # --------------------------------------------------------
    # Figure 1: Position trajectories
    # --------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)

    for i in range(N_AGENTS):
        ax1.plot(
            t,
            X[i, :],
            color=IEEE_AGENT_COLORS[i % len(IEEE_AGENT_COLORS)],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            linewidth=2.4,
            label=rf"$x_{i+1}$"
        )

    ax1.axhline(
        0.0,
        color=IEEE_COLORS["black"],
        linestyle="--",
        linewidth=1.3,
        label=r"Reference"
    )

    format_ieee_axes(
        ax1,
        xlabel=r"Time [s]",
        ylabel=r"Position $x_i(t)$"
    )
    add_ieee_legend(ax1, loc="best", ncol=2)
    save_ieee_figure(fig1, "position_trajectories")
    plt.close(fig1)

    # --------------------------------------------------------
    # Figure 2: Velocity trajectories
    # --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)

    for i in range(N_AGENTS):
        ax2.plot(
            t,
            V[i, :],
            color=IEEE_AGENT_COLORS[i % len(IEEE_AGENT_COLORS)],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            linewidth=2.4,
            label=rf"$v_{i+1}$"
        )

    ax2.axhline(
        0.0,
        color=IEEE_COLORS["black"],
        linestyle="--",
        linewidth=1.3
    )

    format_ieee_axes(
        ax2,
        xlabel=r"Time [s]",
        ylabel=r"Velocity $v_i(t)$"
    )
    add_ieee_legend(ax2, loc="best", ncol=2)
    save_ieee_figure(fig2, "velocity_trajectories")
    plt.close(fig2)

    # --------------------------------------------------------
    # Figure 3: Control signals
    # --------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)

    for i in range(N_AGENTS):
        ax3.plot(
            t,
            U[i, :],
            color=IEEE_AGENT_COLORS[i % len(IEEE_AGENT_COLORS)],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            linewidth=2.4,
            label=rf"$u_{i+1}$"
        )

    ax3.axhline(
        0.0,
        color=IEEE_COLORS["black"],
        linestyle="--",
        linewidth=1.3
    )

    format_ieee_axes(
        ax3,
        xlabel=r"Time [s]",
        ylabel=r"Control input $u_i(t)$"
    )
    add_ieee_legend(ax3, loc="best", ncol=2)
    save_ieee_figure(fig3, "control_signals")
    plt.close(fig3)

    # --------------------------------------------------------
    # Figure 4: Consensus error history
    # --------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax4.semilogy(
        redesign_times,
        cons_err_hist,
        color=IEEE_COLORS["blue"],
        marker="o",
        linewidth=2.4,
        markersize=6.5,
        label=r"$e_c(t_k)$"
    )

    ax4.axhline(
        CONS_TOL,
        color=IEEE_COLORS["black"],
        linestyle="--",
        linewidth=1.3,
        label=r"Tolerance"
    )

    format_ieee_axes(
        ax4,
        xlabel=r"Decision time [s]",
        ylabel=r"Consensus error"
    )
    add_ieee_legend(ax4, loc="best")
    save_ieee_figure(fig4, "consensus_error_history")
    plt.close(fig4)

    # --------------------------------------------------------
    # Figure 5: Controller-parameter redesign
    # --------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax5.step(
        redesign_times,
        Kp_hist,
        where="post",
        color=IEEE_COLORS["blue"],
        linewidth=2.4,
        label=r"$K_p$"
    )
    ax5.plot(
        redesign_times,
        Kp_hist,
        linestyle="None",
        marker="o",
        color=IEEE_COLORS["blue"],
        markersize=6.5
    )

    ax5.step(
        redesign_times,
        Kd_hist,
        where="post",
        color=IEEE_COLORS["orange"],
        linewidth=2.4,
        label=r"$K_d$"
    )
    ax5.plot(
        redesign_times,
        Kd_hist,
        linestyle="None",
        marker="s",
        color=IEEE_COLORS["orange"],
        markersize=6.5
    )

    format_ieee_axes(
        ax5,
        xlabel=r"Decision time [s]",
        ylabel=r"Controller parameters"
    )
    add_ieee_legend(ax5, loc="best", ncol=2)
    save_ieee_figure(fig5, "controller_parameter_redesign")
    plt.close(fig5)

    # --------------------------------------------------------
    # Figure 6: Lyapunov-parameter redesign
    # --------------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax6.step(
        redesign_times,
        theta_x2_hist,
        where="post",
        color=IEEE_COLORS["green"],
        linewidth=2.4,
        label=r"$\theta_{x2}$"
    )
    ax6.plot(
        redesign_times,
        theta_x2_hist,
        linestyle="None",
        marker="o",
        color=IEEE_COLORS["green"],
        markersize=6.5
    )

    ax6.step(
        redesign_times,
        theta_v2_hist,
        where="post",
        color=IEEE_COLORS["purple"],
        linewidth=2.4,
        label=r"$\theta_{v2}$"
    )
    ax6.plot(
        redesign_times,
        theta_v2_hist,
        linestyle="None",
        marker="s",
        color=IEEE_COLORS["purple"],
        markersize=6.5
    )

    ax6.step(
        redesign_times,
        theta_x4_hist,
        where="post",
        color=IEEE_COLORS["yellow"],
        linewidth=2.4,
        label=r"$\theta_{x4}$"
    )
    ax6.plot(
        redesign_times,
        theta_x4_hist,
        linestyle="None",
        marker="^",
        color=IEEE_COLORS["yellow"],
        markersize=6.5
    )

    format_ieee_axes(
        ax6,
        xlabel=r"Decision time [s]",
        ylabel=r"Lyapunov parameters"
    )
    add_ieee_legend(ax6, loc="best", ncol=2)
    save_ieee_figure(fig6, "lyapunov_parameter_redesign")
    plt.close(fig6)

    # --------------------------------------------------------
    # Figure 7: Short-horizon cost history
    # --------------------------------------------------------
    fig7, ax7 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax7.plot(
        redesign_times,
        cost_hist,
        color=IEEE_COLORS["blue"],
        marker="o",
        linewidth=2.4,
        markersize=6.5,
        label=r"$J_k^\star$"
    )

    format_ieee_axes(
        ax7,
        xlabel=r"Decision time [s]",
        ylabel=r"Short-horizon cost"
    )
    add_ieee_legend(ax7, loc="best")
    save_ieee_figure(fig7, "short_horizon_cost_history")
    plt.close(fig7)


# ============================================================
# 8) Run and plot
# ============================================================

if __name__ == "__main__":

    (
        t,
        Z,
        U,
        Kp_hist,
        Kd_hist,
        theta_x2_hist,
        theta_v2_hist,
        theta_x4_hist,
        cost_hist,
        cons_err_hist,
        redesign_times,
        p_min_final,
        p_max_final
    ) = dynamic_quantum_lyapunov_synthesis_with_repeated_BH()

    plot_and_save_all_results(
        t,
        Z,
        U,
        Kp_hist,
        Kd_hist,
        theta_x2_hist,
        theta_v2_hist,
        theta_x4_hist,
        cost_hist,
        cons_err_hist,
        redesign_times
    )

    X = Z[:N_AGENTS, :]
    V = Z[N_AGENTS:, :]

    print("\nSaved TCST-quality PDF figures:")
    print("  1) position_trajectories.pdf")
    print("  2) velocity_trajectories.pdf")
    print("  3) control_signals.pdf")
    print("  4) consensus_error_history.pdf")
    print("  5) controller_parameter_redesign.pdf")
    print("  6) lyapunov_parameter_redesign.pdf")
    print("  7) short_horizon_cost_history.pdf")

    print("\nFinal states at t = {:.4f}:".format(t[-1]))
    for i in range(N_AGENTS):
        print(
            f"  Agent {i+1}: "
            f"x_{i+1}(T) = {X[i, -1]:.6e}, "
            f"v_{i+1}(T) = {V[i, -1]:.6e}"
        )

    x_end = X[:, -1]
    v_end = V[:, -1]

    cons_err_final = np.sqrt(
        np.linalg.norm(L @ x_end, 2)**2
        + np.linalg.norm(L @ v_end, 2)**2
    )

    print("\nFinal consensus error:")
    print("  sqrt(||Lx(T)||_2^2 + ||Lv(T)||_2^2) =", cons_err_final)

    print("\nFinal parameter ranges after last BH:")
    for name, mn, mx in zip(PARAM_NAMES, p_min_final, p_max_final):
        print(f"  {name}: [{mn:.3f}, {mx:.3f}]")