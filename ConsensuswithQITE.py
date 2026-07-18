import numpy as np
from itertools import product
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

# Initial condition for the MAS
X0_GLOBAL = np.array([2.0, -2.5, 3.8, -3.2, 0.3])

# ----- Lyapunov-aware cost -----
T_HORIZON_COST = 0.25
N_TIME_COST = 150
W_Z = 1.0
W_U = 0.1
W_LYAP = 1.0

# ----- Global parameter ranges -----
PARAM_NAMES = ["alpha", "beta", "k", "theta2", "theta4"]
P_MIN_GLOBAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
P_MAX_GLOBAL = np.array([50.0, 2.0, 50.0, 25.0, 25.0])

# ----- Stage 1: Black Hole metaheuristic -----
BH_POP_SIZE = 20
BH_MAX_ITERS = 200
BH_FREEZE_WIDTH = 5.0

# ----- Stage 2: Dynamic quantum Lyapunov synthesis -----
T_TOTAL = 10.0
DT_DECISION = 0.25
QITE_TAU = 3.0
QITE_STEPS = 60
QITE_REPS = 2
QITE_SEED_BASE = 42

# Stopping criterion for consensus
CONS_TOL = 1e-8

# Hamiltonian fitting and QITE candidate settings
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

def closed_loop_dynamics(t, x, alpha, beta, k_gain):
    """
    Closed-loop dynamics for the scalar MAS:

        dot x_i = x_i + x_i^3 + u_i
        u_i = -alpha*x_i - beta*x_i^3 - k * sum_{j in N_i} (x_i - x_j)

    Therefore:

        dot x = (1-alpha)*x + (1-beta)*x^3 - k L x
    """
    x = np.asarray(x)
    local_drift = x + x**3
    control_u = -alpha * x - beta * x**3 - k_gain * (L @ x)
    dxdt = local_drift + control_u
    return dxdt


def control_law(x, alpha, beta, k_gain):
    """
    Compute control input u(t) for a given state x and gains.
    x can be shape (N,) or (N, T).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return -alpha * x - beta * x**3 - k_gain * (L @ x)
    else:
        return -alpha * x - beta * x**3 - k_gain * (L @ x)


def lyapunov_V(x, theta2, theta4):
    """
    V(x) = (theta2/2) * ||x||^2 + (theta4/4) * sum_i x_i^4
    """
    x = np.asarray(x)
    return 0.5 * theta2 * np.sum(x**2) + 0.25 * theta4 * np.sum(x**4)


def lyapunov_dVdt(x, dxdt, theta2, theta4):
    """
    dV/dt = sum_i (dV/dx_i * dx_i/dt)
    where dV/dx_i = theta2*x_i + theta4*x_i^3.
    """
    x = np.asarray(x)
    dxdt = np.asarray(dxdt)
    dVdx = theta2 * x + theta4 * x**3
    return float(np.dot(dVdx, dxdt))


def simulate_horizon_cost(x0,
                          alpha, beta, k_gain, theta2, theta4,
                          t_horizon=T_HORIZON_COST,
                          n_time=N_TIME_COST,
                          w_z=W_Z,
                          w_u=W_U,
                          w_lyap=W_LYAP):
    """
    Short-horizon Lyapunov-aware cost from x0.
    This is the core black-box cost used in both BH and QITE selection.
    """
    t_span = (0.0, t_horizon)
    t_eval = np.linspace(t_span[0], t_span[1], n_time)

    def dyn(t, x):
        return closed_loop_dynamics(t, x, alpha, beta, k_gain)

    sol = solve_ivp(
        dyn,
        t_span,
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        return 1e6

    t = sol.t
    X = sol.y

    # Blow-up check
    if np.any(np.abs(X) > 1e3):
        return 1e6

    Z = L @ X
    U = control_law(X, alpha, beta, k_gain)

    z_norm_sq = np.sum(Z**2, axis=0)
    u_norm_sq = np.sum(U**2, axis=0)
    integrand_perf = w_z * z_norm_sq + w_u * u_norm_sq

    dVdt_vals = []
    for idx, tt in enumerate(t):
        x_t = X[:, idx]
        dxdt_t = dyn(tt, x_t)
        dVdt_vals.append(lyapunov_dVdt(x_t, dxdt_t, theta2, theta4))

    dVdt_vals = np.array(dVdt_vals)

    lyap_violation = np.maximum(0.0, dVdt_vals)
    integrand_lyap = lyap_violation**2

    J_perf = float(np.trapz(integrand_perf, t))
    J_lyap = float(np.trapz(integrand_lyap, t))
    J_total = J_perf + w_lyap * J_lyap

    return J_total


# ============================================================
# 2) Stage 1: Black Hole range calibration
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


def bh_evaluate_population(stars, x0):
    """
    Evaluate black-box cost for each star.
    """
    costs = np.zeros(len(stars))
    for i, p in enumerate(stars):
        alpha, beta, k_gain, theta2, theta4 = p
        costs[i] = simulate_horizon_cost(x0, alpha, beta, k_gain, theta2, theta4)
    return costs


def bh_range_calibration(p_min_init, p_max_init, x0):
    """
    Stage 1: simplified Black Hole metaheuristic to shrink parameter ranges
    around the current state x0 at a given decision step.
    """
    np.random.seed(0)

    n_params = len(PARAM_NAMES)
    stars = bh_initialize_population(p_min_init, p_max_init)
    active = np.ones(n_params, dtype=bool)
    p_min = p_min_init.copy()
    p_max = p_max_init.copy()

    for it in range(BH_MAX_ITERS):
        costs = bh_evaluate_population(stars, x0)
        best_idx = np.argmin(costs)
        black_hole = stars[best_idx].copy()
        best_cost = costs[best_idx]

        if VERBOSE:
            print(f"[BH] Iter {it+1}/{BH_MAX_ITERS}, best_cost={best_cost:.4e}")

        # Move stars toward black hole on active coordinates
        for i in range(BH_POP_SIZE):
            if i == best_idx:
                continue

            rand_vec = np.random.rand(np.sum(active))
            stars[i, active] = (
                stars[i, active]
                + rand_vec * (black_hole[active] - stars[i, active])
            )

        # Clip to current ranges
        for j in range(n_params):
            stars[:, j] = np.clip(stars[:, j], p_min[j], p_max[j])

        # Update ranges
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
# 3) Bit allocation and parameter encoding
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
        print("[Bits] Allocation per param:")
        for name, w, b in zip(PARAM_NAMES, widths, bits_per_param):
            print(f"   {name}: width={w:.3f}, bits={b}")
        print(f"   Total qubits = {num_qubits}")

    return bits_per_param, num_qubits


def decode_bitstring_to_params(bitstr, p_min, p_max, bits_per_param):
    """
    Decode a bitstring to continuous parameters.
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
# 4) Diagonal Pauli basis and Hamiltonian fitting
# ============================================================

def build_diagonal_pauli_basis(num_qubits):
    """
    Diagonal Pauli basis:
        {I, Z_i, Z_i Z_j}
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
    Eigenvalue of a diagonal Pauli operator on a computational-basis bitstring.
    """
    n = len(bitstr)
    assert len(pauli_label) == n

    bits = [int(b) for b in bitstr]
    eig = 1.0

    # Qiskit convention: rightmost bit is qubit 0
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
    bitstrings = set()
    max_possible = 2**num_qubits
    num_samples = min(num_samples, max_possible)

    while len(bitstrings) < num_samples:
        s = "".join(np.random.choice(["0", "1"], size=num_qubits))
        bitstrings.add(s)

    return list(bitstrings)


def fit_diagonal_hamiltonian_sampled(x0, p_min, p_max, bits_per_param,
                                     paulis, num_qubits):
    """
    Fit H = sum_k h_k P_k using sampled bitstrings.
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
        alpha, beta, k_gain, theta2, theta4 = decode_bitstring_to_params(
            s, p_min, p_max, bits_per_param
        )

        J = simulate_horizon_cost(x0, alpha, beta, k_gain, theta2, theta4)
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

def qite_select_best_bitstring(H, x0, p_min, p_max, bits_per_param,
                               tau=QITE_TAU,
                               steps=QITE_STEPS,
                               reps=QITE_REPS,
                               seed=0):
    """
    Run VarQITE on H, obtain final state, and select the best bitstring
    according to the true black-box cost.
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
        print("[QITE] Evaluating true cost on top-probability candidates:")

    for s, p_prob in candidates:
        alpha, beta, k_gain, theta2, theta4 = decode_bitstring_to_params(
            s, p_min, p_max, bits_per_param
        )
        J = simulate_horizon_cost(x0, alpha, beta, k_gain, theta2, theta4)

        if VERBOSE:
            print(f"   s={s}, p_prob={p_prob:.4f}, J={J:.4f}")

        if J < best_J:
            best_s, best_J = s, J

    energy = est.run([final_circ], [H]).result().values[0]

    if VERBOSE:
        print(f"[QITE] Best bitstring {best_s} with J={best_J:.4f}, <H>={energy:.4f}")

    return best_s, best_J


# ============================================================
# 6) Full loop: BH + dynamic QITE + early stopping
# ============================================================

def dynamic_quantum_lyapunov_synthesis_with_repeated_BH():
    """
    Full algorithm with additional histories for publication plots:

      1. Agent-state trajectories
      2. Control signals
      3. Consensus error history
      4. Controller-parameter redesign history
      5. Lyapunov-parameter redesign history
      6. Short-horizon cost history
    """
    print("=== Dynamic Quantum Lyapunov Synthesis with BH at Every Step ===")

    p_min_curr = P_MIN_GLOBAL.copy()
    p_max_curr = P_MAX_GLOBAL.copy()

    x_current = X0_GLOBAL.copy()

    t_global = [0.0]
    X_global = [x_current.copy()]
    U_global = [control_law(x_current, 0.0, 0.0, 0.0)]

    # Controller-parameter histories
    alpha_hist = []
    beta_hist = []
    k_hist = []

    # Lyapunov-parameter histories
    theta2_hist = []
    theta4_hist = []

    # Histories for publication plots
    cost_hist = []
    cons_err_hist = []
    redesign_times = []

    max_steps = int(T_TOTAL / DT_DECISION)

    for k_step in range(max_steps):
        print(f"\n=== Decision step {k_step+1}/{max_steps} ===")

        if VERBOSE:
            print(f"   Current state x_k = {x_current}")

        # ---------------------------------------------------
        # Stage 1: BH range calibration
        # ---------------------------------------------------
        p_min_step, p_max_step = bh_range_calibration(
            p_min_curr,
            p_max_curr,
            x_current
        )
        p_min_curr = p_min_step
        p_max_curr = p_max_step

        print("   Updated parameter ranges after BH:")
        for name, mn, mx in zip(PARAM_NAMES, p_min_curr, p_max_curr):
            print(f"      {name}: [{mn:.3f}, {mx:.3f}] (width={mx-mn:.3f})")

        # ---------------------------------------------------
        # Bit allocation and Hamiltonian construction
        # ---------------------------------------------------
        bits_per_param, num_qubits = allocate_bits_for_parameters(
            p_min_curr,
            p_max_curr
        )
        paulis = build_diagonal_pauli_basis(num_qubits)

        H = fit_diagonal_hamiltonian_sampled(
            x_current,
            p_min_curr,
            p_max_curr,
            bits_per_param,
            paulis,
            num_qubits
        )

        # ---------------------------------------------------
        # Stage 2: QITE-based parameter selection
        # ---------------------------------------------------
        best_s, best_J = qite_select_best_bitstring(
            H,
            x_current,
            p_min_curr,
            p_max_curr,
            bits_per_param,
            tau=QITE_TAU,
            steps=QITE_STEPS,
            reps=QITE_REPS,
            seed=QITE_SEED_BASE + k_step
        )

        alpha, beta, k_gain, theta2, theta4 = decode_bitstring_to_params(
            best_s,
            p_min_curr,
            p_max_curr,
            bits_per_param
        )

        print(
            f"   Selected params: s={best_s}, "
            f"alpha={alpha:.3f}, beta={beta:.3f}, k={k_gain:.3f}, "
            f"theta2={theta2:.3f}, theta4={theta4:.3f}, J={best_J:.4f}"
        )

        # Store redesign history at the current decision step
        alpha_hist.append(alpha)
        beta_hist.append(beta)
        k_hist.append(k_gain)
        theta2_hist.append(theta2)
        theta4_hist.append(theta4)
        cost_hist.append(best_J)

        # ---------------------------------------------------
        # Integrate true closed-loop system over one decision interval
        # ---------------------------------------------------
        t_start = t_global[-1]
        t_end = t_start + DT_DECISION
        redesign_times.append(t_end)

        t_span = (t_start, t_end)
        t_eval = np.linspace(t_start, t_end, 200)

        def dyn_interval(t, x):
            return closed_loop_dynamics(t, x, alpha, beta, k_gain)

        sol = solve_ivp(
            dyn_interval,
            t_span,
            x_current,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )

        if not sol.success:
            print("   Integration failed in interval:", sol.message)
            break

        t_local = sol.t
        X_local = sol.y
        U_local = control_law(X_local, alpha, beta, k_gain)

        if k_step == 0 and len(t_global) == 1:
            t_global = list(t_local)
            X_global = [X_local[:, i] for i in range(X_local.shape[1])]
            U_global = [U_local[:, i] for i in range(U_local.shape[1])]
        else:
            t_global.extend(list(t_local[1:]))
            X_global.extend([X_local[:, i] for i in range(1, X_local.shape[1])])
            U_global.extend([U_local[:, i] for i in range(1, U_local.shape[1])])

        x_current = X_local[:, -1].copy()
        cons_err = np.linalg.norm(L @ x_current, 2)
        cons_err_hist.append(cons_err)

        print(f"   State at end of interval: {x_current}")
        print(f"   Consensus error ||Lx||_2 = {cons_err:.3e}")

        if cons_err <= CONS_TOL:
            print("   Consensus error below tolerance; stopping simulation.")
            break

    # Convert global trajectories and histories to arrays
    t_global = np.array(t_global)
    X_global = np.array(X_global).T
    U_global = np.array(U_global).T

    alpha_hist = np.array(alpha_hist)
    beta_hist = np.array(beta_hist)
    k_hist = np.array(k_hist)
    theta2_hist = np.array(theta2_hist)
    theta4_hist = np.array(theta4_hist)
    cost_hist = np.array(cost_hist)
    cons_err_hist = np.array(cons_err_hist)
    redesign_times = np.array(redesign_times)

    return (
        t_global,
        X_global,
        U_global,
        alpha_hist,
        beta_hist,
        k_hist,
        theta2_hist,
        theta4_hist,
        cost_hist,
        cons_err_hist,
        redesign_times,
        p_min_curr,
        p_max_curr
    )


# ============================================================
# 7) Plotting function for all requested TCST-style figures
# ============================================================

def plot_and_save_all_results(t, X, U,
                              alpha_hist, beta_hist, k_hist,
                              theta2_hist, theta4_hist,
                              cost_hist, cons_err_hist,
                              redesign_times):
    """
    Generate and save all TCST-style figures:

      1. Agent-state trajectories
      2. Control signals
      3. Consensus error history
      4. Controller-parameter redesign
      5. Lyapunov-parameter redesign
      6. Short-horizon cost history
    """

    configure_ieee_tcst_plot_style()

    # --------------------------------------------------------
    # Figure 1: Agent-state trajectories
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
        ylabel=r"State $x_i(t)$"
    )
    add_ieee_legend(ax1, loc="best", ncol=2)
    save_ieee_figure(fig1, "mas_states")
    plt.close(fig1)

    # --------------------------------------------------------
    # Figure 2: Control signals
    # --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)

    for i in range(N_AGENTS):
        ax2.plot(
            t,
            U[i, :],
            color=IEEE_AGENT_COLORS[i % len(IEEE_AGENT_COLORS)],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            linewidth=2.4,
            label=rf"$u_{i+1}$"
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
        ylabel=r"Control input $u_i(t)$"
    )
    add_ieee_legend(ax2, loc="best", ncol=2)
    save_ieee_figure(fig2, "mas_control")
    plt.close(fig2)

    # --------------------------------------------------------
    # Figure 3: Consensus error history
    # --------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax3.semilogy(
        redesign_times,
        cons_err_hist,
        color=IEEE_COLORS["blue"],
        marker="o",
        linewidth=2.4,
        markersize=6.5,
        label=r"$\|Lx(t_k)\|_2$"
    )

    ax3.axhline(
        CONS_TOL,
        color=IEEE_COLORS["black"],
        linestyle="--",
        linewidth=1.3,
        label=r"Tolerance"
    )

    format_ieee_axes(
        ax3,
        xlabel=r"Decision time [s]",
        ylabel=r"Consensus error"
    )
    add_ieee_legend(ax3, loc="best")
    save_ieee_figure(fig3, "consensus_error_history")
    plt.close(fig3)

    # --------------------------------------------------------
    # Figure 4: Controller-parameter redesign
    # --------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax4.step(
        redesign_times,
        alpha_hist,
        where="post",
        color=IEEE_COLORS["blue"],
        linewidth=2.4,
        label=r"$\alpha$"
    )
    ax4.plot(
        redesign_times,
        alpha_hist,
        linestyle="None",
        marker="o",
        color=IEEE_COLORS["blue"],
        markersize=6.5
    )

    ax4.step(
        redesign_times,
        beta_hist,
        where="post",
        color=IEEE_COLORS["orange"],
        linewidth=2.4,
        label=r"$\beta$"
    )
    ax4.plot(
        redesign_times,
        beta_hist,
        linestyle="None",
        marker="s",
        color=IEEE_COLORS["orange"],
        markersize=6.5
    )

    ax4.step(
        redesign_times,
        k_hist,
        where="post",
        color=IEEE_COLORS["green"],
        linewidth=2.4,
        label=r"$k$"
    )
    ax4.plot(
        redesign_times,
        k_hist,
        linestyle="None",
        marker="^",
        color=IEEE_COLORS["green"],
        markersize=6.5
    )

    format_ieee_axes(
        ax4,
        xlabel=r"Decision time [s]",
        ylabel=r"Controller parameters"
    )
    add_ieee_legend(ax4, loc="best", ncol=3)
    save_ieee_figure(fig4, "controller_parameter_redesign")
    plt.close(fig4)

    # --------------------------------------------------------
    # Figure 5: Lyapunov-parameter redesign
    # --------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax5.step(
        redesign_times,
        theta2_hist,
        where="post",
        color=IEEE_COLORS["purple"],
        linewidth=2.4,
        label=r"$\theta_2$"
    )
    ax5.plot(
        redesign_times,
        theta2_hist,
        linestyle="None",
        marker="o",
        color=IEEE_COLORS["purple"],
        markersize=6.5
    )

    ax5.step(
        redesign_times,
        theta4_hist,
        where="post",
        color=IEEE_COLORS["yellow"],
        linewidth=2.4,
        label=r"$\theta_4$"
    )
    ax5.plot(
        redesign_times,
        theta4_hist,
        linestyle="None",
        marker="s",
        color=IEEE_COLORS["yellow"],
        markersize=6.5
    )

    format_ieee_axes(
        ax5,
        xlabel=r"Decision time [s]",
        ylabel=r"Lyapunov parameters"
    )
    add_ieee_legend(ax5, loc="best", ncol=2)
    save_ieee_figure(fig5, "lyapunov_parameter_redesign")
    plt.close(fig5)

    # --------------------------------------------------------
    # Figure 6: Short-horizon cost history
    # --------------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    ax6.plot(
        redesign_times,
        cost_hist,
        color=IEEE_COLORS["blue"],
        marker="o",
        linewidth=2.4,
        markersize=6.5,
        label=r"$J_k^\star$"
    )

    format_ieee_axes(
        ax6,
        xlabel=r"Decision time [s]",
        ylabel=r"Short-horizon cost"
    )
    add_ieee_legend(ax6, loc="best")
    save_ieee_figure(fig6, "short_horizon_cost_history")
    plt.close(fig6)


# ============================================================
# 8) Run and save all plots
# ============================================================

if __name__ == "__main__":

    (
        t,
        X,
        U,
        alpha_hist,
        beta_hist,
        k_hist,
        theta2_hist,
        theta4_hist,
        cost_hist,
        cons_err_hist,
        redesign_times,
        p_min_final,
        p_max_final
    ) = dynamic_quantum_lyapunov_synthesis_with_repeated_BH()

    plot_and_save_all_results(
        t,
        X,
        U,
        alpha_hist,
        beta_hist,
        k_hist,
        theta2_hist,
        theta4_hist,
        cost_hist,
        cons_err_hist,
        redesign_times
    )

    print("\nSaved TCST-quality PDF figures:")
    print("  1) mas_states.pdf")
    print("  2) mas_control.pdf")
    print("  3) consensus_error_history.pdf")
    print("  4) controller_parameter_redesign.pdf")
    print("  5) lyapunov_parameter_redesign.pdf")
    print("  6) short_horizon_cost_history.pdf")

    print("\nFinal states at t = {:.4f}:".format(t[-1]))
    for i in range(N_AGENTS):
        print(f"  Agent {i+1}: x_{i+1}(T) = {X[i, -1]:.6e}")

    print("\nFinal consensus error:")
    print("  ||Lx(T)||_2 =", np.linalg.norm(L @ X[:, -1], 2))

    print("\nFinal parameter ranges after last BH:")
    for name, mn, mx in zip(PARAM_NAMES, p_min_final, p_max_final):
        print(f"  {name}: [{mn:.3f}, {mx:.3f}]")