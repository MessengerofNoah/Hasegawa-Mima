import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, linalg

# Parameters
L = 2*np.pi
N = 128
dx = L / N
dy = L / N
dt = 0.01
tmax = 10.0
v_star = 2.0

x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)
output_times = [0.0, tmax/3, 2*tmax/3, tmax]

# Initial condition
phi0 = np.exp(-((X - L/2)**2 + (Y - L/2)**2)/0.2**2)

# -------- Method 1: FFT Spectral Method --------
phi_hat0 = np.fft.fft2(phi0)
kx = np.fft.fftfreq(N, d=dx)*2*np.pi
ky = np.fft.fftfreq(N, d=dy)*2*np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2
A_fft = -(1 + k2)

snapshots_fft = []
for t in output_times:
    phase = np.exp(-1j * v_star * KY * t / A_fft)
    phi_hat_t = phi_hat0 * phase
    phi_t = np.real(np.fft.ifft2(phi_hat_t))
    snapshots_fft.append(phi_t)

# -------- Method 2: Sparse Matrix Iterative Inversion --------
# Construct Helmholtz operator
e = np.ones(N)
L1D = diags([e, -2*e, e], offsets=[-1,0,1], shape=(N,N)).tolil()
L1D[0,-1] = 1
L1D[-1,0] = 1
L1D /= dx**2
L2D = kron(eye(N), L1D) + kron(L1D, eye(N))
A_sparse = L2D - eye(N*N)

phi_vec = phi0.ravel()
q_vec = A_sparse @ phi_vec

t = 0.0
nsteps = int(tmax/dt)
snapshots_sparse = []
output_idx = 0

while t <= tmax + 1e-8:
    phi_grid = phi_vec.reshape((N,N))
    dphi_dy = np.roll(phi_grid, -1, axis=0) - np.roll(phi_grid, +1, axis=0)
    dphi_dy /= (2*dy)
    rhs = -v_star * dphi_dy.ravel()
    q_vec = q_vec + dt*rhs
    phi_vec, info = linalg.cg(A_sparse, q_vec, atol=1e-8, maxiter=500)
    if np.isclose(t, output_times[output_idx], atol=dt/2):
        snapshots_sparse.append(phi_vec.reshape((N,N)).copy())
        output_idx += 1
        if output_idx >= len(output_times):
            break
    t += dt

# -------- Error Calculation and Visualization --------
vmin = min(np.min(s) for s in snapshots_fft + snapshots_sparse)
vmax = max(np.max(s) for s in snapshots_fft + snapshots_sparse)

l2_norms = []
linf_norms = []

plt.figure(figsize=(12, 4*len(output_times)))
for idx, t in enumerate(output_times):
    phi_fft = snapshots_fft[idx]
    phi_sparse = snapshots_sparse[idx]
    diff = phi_fft - phi_sparse
    l2 = np.sqrt(np.mean(diff**2))
    linf = np.max(np.abs(diff))
    l2_norms.append(l2)
    linf_norms.append(linf)

    plt.subplot(len(output_times), 3, idx*3+1)
    plt.pcolormesh(X, Y, phi_fft, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'FFT Spectral t={t:.2f}')
    plt.axis('off')

    plt.subplot(len(output_times), 3, idx*3+2)
    plt.pcolormesh(X, Y, phi_sparse, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Sparse Inversion t={t:.2f}')
    plt.axis('off')

    plt.subplot(len(output_times), 3, idx*3+3)
    plt.pcolormesh(X, Y, diff, cmap='bwr', shading='auto')
    plt.title(f'Difference\nL2={l2:.2e}, L∞={linf:.2e}')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("L2 norms at each time:", l2_norms)
print("L-infinity norms at each time:", linf_norms)
print(f"Mean L2 norm: {np.mean(l2_norms):.6e}")
print(f"Mean L-infinity norm: {np.mean(linf_norms):.6e}")