import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = './logs/local/2026-11-06_18-01-48_988528/eval/samples/'

dict_energies = {
    'ckpt_id': [],
    'avg_energy': []
}
for ckpt_id in range(100, 5100, 100):
    df = pd.read_csv(BASE_DIR + f'gfn_samples_ckpt_{ckpt_id}.csv')
    energies = df['energies']

    dict_energies['ckpt_id'].append(ckpt_id)
    dict_energies['avg_energy'].append(energies.mean())


df_energies = pd.DataFrame(dict_energies)
sns.lineplot(df_energies, x='ckpt_id', y='avg_energy', marker="o")
plt.xlabel('Timestep', fontsize=18)
plt.ylabel('Average reward (normalized)', fontsize=18)
# plt.show()
plt.savefig(f"training_energies.png", dpi=300, bbox_inches="tight")