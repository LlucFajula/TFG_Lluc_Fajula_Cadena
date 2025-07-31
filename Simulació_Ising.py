import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os


class IsingGrid:
    def __init__(self, N, J, h_init):
        self.N = N
        self.J = J          
        self.h = h_init.copy()  
        self.spins = np.random.choice([-1, 1], size=(N, N))

    def get_energy(self):
        """calcula energia total de la configuracio actual de spins"""
        e = 0
        for i in range(self.N):
            for j in range(self.N):
                s = self.spins[i, j]
                neigh = (self.spins[(i+1)%self.N, j] +
                         self.spins[(i-1)%self.N, j] +
                         self.spins[i, (j+1)%self.N] +
                         self.spins[i, (j-1)%self.N])
                e += -self.J * s * neigh - self.h[i, j] * s
        return e / 2  # cada parell comptat dues vegades

    def metropolis_step(self, T):
        """intent de flip aleatori d'un spin"""
        i, j = np.random.randint(0, self.N, size=2)
        current = self.spins[i, j]
        flipped = -current
        neigh = (self.spins[(i+1)%self.N, j] +
                 self.spins[(i-1)%self.N, j] +
                 self.spins[i, (j+1)%self.N] +
                 self.spins[i, (j-1)%self.N])
        delta_E = - self.J * (flipped - current) * neigh \
                  - self.h[i, j] * (flipped - current)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            self.spins[i, j] = flipped

    def simulate(self, T, h_grid, num_steps, vis_interval=None):
        """simula num_steps passos. Retorna llistes de energies, magnetitzacions i configs intermitges"""
        energies, mags, configs = [], [], []
        for t in range(num_steps):
            self.h = h_grid[t]
            self.metropolis_step(T)
            energies.append(self.get_energy())
            mags.append(self.spins.sum())
            if vis_interval and (t % vis_interval == 0 or t == num_steps-1):
                configs.append({
                    't': t,
                    'h': h_grid[t].copy(),
                    'spins': self.spins.copy()
                })
        return energies, mags, configs


# Parametres generals i lectura de dades

N, J, T_model    = 35, 0.4, 2.0
vis_interval     = 3600          # cada 3600 passos guardem frame
seconds_per_hour = 3600

beta_calib = 2.53           # factor d'escalat per convertir la magnetització

#temperatures horàries
temp_file = r"D:\Física\TFG\Dades\FINALS\temperatures_grid_35x35_iberia_BONES.xlsx"
df_t = pd.read_excel(temp_file, index_col=0)
df_t.index = pd.to_datetime(df_t.index).tz_localize(None)

# diferències de consum 
diff_file = r"D:\Física\TFG\Dades\FINALS\consum_diferencia_35x35_iberia.xlsx"
df_diff = pd.read_excel(diff_file, parse_dates=['temps'], index_col='temps')
df_diff.index = df_diff.index.tz_localize(None)

# matriu de temperatures
expected_cols = [f"T_{i}_{j}" for i in range(N) for j in range(N)]
arr_hourly    = df_t[expected_cols].values.reshape(-1, N, N)
#Upsample
h_hourly_temp = np.repeat(arr_hourly, seconds_per_hour, axis=0)
num_steps     = h_hourly_temp.shape[0]


#loop alphas
alphas = []
for alpha in alphas:
   
    #    h(t) = alpha*(T(t)−T0)·cos(2pi x t_h/12 + phi) + lambda sin(2pi x t_h/168)
    T0 = 14.1

    n_hours    = arr_hourly.shape[0]
    hours      = np.arange(n_hours)
    cos_hourly = np.cos(2*np.pi * hours / 12 + np.pi/3 )
    sin_hourly = 0.20* np.sin(2*np.pi * hours / 168 + np.pi)
    cos_full = np.repeat(cos_hourly, seconds_per_hour)
    sin_full = np.repeat(sin_hourly, seconds_per_hour)

    h_grid_real = (alpha* (h_hourly_temp - T0) * cos_full[:, None, None] + sin_full[:, None, None] )

    #  Simula
    ising = IsingGrid(N, J, h_grid_real[0])
    energies, mags, configs = ising.simulate(
        T_model, h_grid_real, num_steps, vis_interval
    )
    print(f"\n Simulació completada per alpha = {alpha}")



    # RMSE sense burnin
    mags_hourly = np.array(mags)[::seconds_per_hour]
    DeltaC_sim  = beta_calib * mags_hourly

    df_model = pd.DataFrame({
        'energy_hourly': energies[::seconds_per_hour],
        'mag_hourly':    mags_hourly,
        'DeltaC_sim':    DeltaC_sim  }, index=df_t.index)

    df_all = df_model.join(
        df_diff[['Consum_real','Consum_previst','diferència']],
        how='inner'
    )
    rmse = np.sqrt(
        np.mean((df_all['DeltaC_sim'] - df_all['diferència'])**2)
    )
    print(f"RMSE de simulació sencera = {rmse:.3f} MW")

    # exportació Excel tots els valors horaris
    out_dir   = r"D:\Física\TFG\Figures\Fetes Servir"
    alpha_tag = f"alpha_{str(alpha).replace('.', '_')}"
    excel_fp  = os.path.join(out_dir, f"resultats_{alpha_tag}.xlsx")
    with pd.ExcelWriter(excel_fp, engine='xlsxwriter') as writer:
        df_all.to_excel(writer, sheet_name='hora_a_hora')
    print("Sèries exportades a", excel_fp)

    # límits de color d'escala per gifs
    temp_min, temp_max = h_hourly_temp.min(),   h_hourly_temp.max()
    h_min,     h_max   = h_grid_real.min(),     h_grid_real.max()

    #rutes de sortida amb etiqueta 
    frames_temp_dir = os.path.join(out_dir, alpha_tag, "frames_temp_h")
    frames_spin_dir = os.path.join(out_dir, alpha_tag, "frames_spins")
    os.makedirs(frames_temp_dir, exist_ok=True)
    os.makedirs(frames_spin_dir, exist_ok=True)

    frame_paths_temp_h = []
    frame_paths_spins   = []
    #Generació de frames gifs
    for entry in configs:
        t          = entry['t']
        temp_frame = h_hourly_temp[t]
        h_frame    = h_grid_real[t]
        # temperatura + camp
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        im0 = axs[0].imshow(temp_frame, cmap='inferno',
                            origin='upper', vmin=temp_min, vmax=temp_max)
        axs[0].set_title(f'T={t}s Temperatura')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(h_frame, cmap='coolwarm',
                            origin='upper', vmin=h_min, vmax=h_max)
        axs[1].set_title(f'T={t}s Camp magnètic')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        path_th = os.path.join(frames_temp_dir, f"frame_{t:06d}.png")
        plt.tight_layout(); fig.savefig(path_th); plt.close(fig)
        frame_paths_temp_h.append(path_th)
        # spins
        fig, ax = plt.subplots(figsize=(4, 4))
        im2 = ax.imshow(entry['spins'], cmap='coolwarm',
                        origin='lower', vmin=-1, vmax=1)
        ax.set_title(f'T={t}s Spins')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        path_sp = os.path.join(frames_spin_dir, f"frame_{t:06d}.png")
        plt.tight_layout(); fig.savefig(path_sp); plt.close(fig)
        frame_paths_spins.append(path_sp)

    gif_temp  = os.path.join(out_dir, alpha_tag, f"evolucio_temp_{alpha_tag}.gif")
    gif_spins = os.path.join(out_dir, alpha_tag, f"evolucio_spins_{alpha_tag}.gif")
    with imageio.get_writer(gif_temp, mode='I', duration=0.2) as writer:
        for fp in frame_paths_temp_h:
            writer.append_data(imageio.imread(fp))
    with imageio.get_writer(gif_spins, mode='I', duration=0.2) as writer:
        for fp in frame_paths_spins:
            writer.append_data(imageio.imread(fp))

    print("gifs creats:", gif_temp, "&", gif_spins)

    #Gràfics energia i magnetització
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(energies)
    ax.set(xlabel="Pas de la simulació (s)", ylabel="Energia")
    ax.grid(True); fig.tight_layout()
    fn1 = os.path.join(out_dir, alpha_tag, f"energy_{alpha_tag}.png")
    fig.savefig(fn1); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(mags, color="tab:green")
    ax.set(xlabel="Pas de la simulació (s)", ylabel="Magnetització")
    ax.grid(True); fig.tight_layout()
    fn2 = os.path.join(out_dir, alpha_tag, f"magnetization_{alpha_tag}.png")
    fig.savefig(fn2); plt.close(fig)
    print("Gràfics d’energia i magnetització:", fn1, "&", fn2)

    # grafics de desviacions
    plt.figure(figsize=(10,4))
    plt.plot(df_all.index, df_all['diferència'], label=r'$\Delta C_{real}$', color='C1')
    plt.plot(df_all.index, df_all['DeltaC_sim'],  label=r'$\Delta C_{sim}$',  color='C2')
    plt.xlabel("Dia de la simulació"); plt.ylabel("Desviació respecte el consum real (MW)")
    #plt.title(f"desviacio real vs sim (α={alpha})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    g1 = os.path.join(out_dir, alpha_tag, f"DeltaC_comp_{alpha_tag}.png")
    plt.savefig(g1); plt.close()
    # grafic consums
    plt.figure(figsize=(10,4))
    total_sim = df_all['Consum_previst'] + df_all['DeltaC_sim']
    plt.plot(df_all.index, df_all['Consum_real'], label=r'$C_{real}$',   color='C0')
    plt.plot(df_all.index, total_sim, label=r'$C_{prev}+\Delta C_{sim}$', color='C3')
    plt.xlabel("Dia de la simulació"); plt.ylabel("Consum (MW)")
    #plt.title(f"Consum real vs previst+sim (alpha={alpha})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    g2 = os.path.join(out_dir, alpha_tag, f"Consums_comp_{alpha_tag}.png")
    plt.savefig(g2); plt.close()
    print("Gràfics comparació guardats:", g1, "&", g2)
