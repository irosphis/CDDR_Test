import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

data = np.loadtxt('Data/cc.txt')
z_cc = data[:, 0]  # 红移
H_data = data[:, 1]  # H(z), 单位：km/s/Mpc
H_err = data[:, 2]  # H(z) 的误差

# BAO D_V 的红移点
z_BAO, _, _ = np.loadtxt('Data/Combine_DV_rd.dat', unpack=True)

# 定义核函数
kernel = C(1.0, (1e-3, 1e10)) * Matern(length_scale=1, length_scale_bounds=(1e-2, 1e3), nu=5/2)

# 高斯过程回归
gpr = GaussianProcessRegressor(kernel=kernel, alpha=H_err**2, n_restarts_optimizer=10)
gpr.fit(z_cc[:, np.newaxis], H_data)

# 定义用于绘图的红移网格
z_grid = np.linspace(0, 2.5, 200)
H_pred_grid, H_std_grid = gpr.predict(z_grid[:, np.newaxis], return_std=True)

H_pred_BAO, H_std_BAO = gpr.predict(z_BAO[:, np.newaxis], return_std=True)

# 打印 BAO 红移点的结果
print("z_BAO:", z_BAO)
print("H(z) at BAO points:", H_pred_BAO)
print("H(z) error at BAO points:", H_std_BAO)


# 保存 BAO 点预测结果
with open("Output/Hz_DVreconstruct_results.txt", "w") as file:
    file.write("# z_DV, H(z), H(z) error\n")
    for z, H, H_err_BAO in zip(z_BAO, H_pred_BAO, H_std_BAO):
        file.write(f"{z} {H} {H_err_BAO}\n")

# 绘图
plt.figure(figsize=(8, 6), dpi=100)

# 绘制原始数据点（带误差棒）
plt.errorbar(z_cc, H_data, yerr=H_err, fmt='o', color='#1f77b4', ecolor='#1f77b4', 
             capsize=3, markersize=5, label='Cosmic Chronometric Data (31 points)')

# 绘制重构的 H(z) 曲线
plt.plot(z_grid, H_pred_grid, color='#d62728', linewidth=1.5, label='GPR Reconstruction')

# 绘制 1σ 和 2σ 置信区间
plt.fill_between(z_grid, H_pred_grid - 3*H_std_grid, H_pred_grid + 3*H_std_grid, 
                 color='#d62728', alpha=0.05, label=r'3$\sigma$ Confidence Interval')
plt.fill_between(z_grid, H_pred_grid - 2*H_std_grid, H_pred_grid + 2*H_std_grid, 
                 color='#d62728', alpha=0.1, label=r'2$\sigma$ Confidence Interval')
plt.fill_between(z_grid, H_pred_grid - H_std_grid, H_pred_grid + H_std_grid, 
                 color='#d62728', alpha=0.3, label=r'1$\sigma$ Confidence Interval')

# 绘制 BAO 红移点的预测值
plt.errorbar(z_BAO, H_pred_BAO, yerr=H_std_BAO, fmt='*', color='#2ca02c', 
             ecolor='#2ca02c', markersize=10, capsize=3, label=r'$H(z)$ at $z_{(D_V/r_d)}$ Points')

# 图表设置
plt.xlabel('Redshift $z$', fontsize=16, fontfamily='serif')
plt.ylabel('$H(z)$ [km/s/Mpc]', fontsize=16, fontfamily='serif')
# plt.title('Reconstructed $H(z)$ from Cosmic Chronometric Data', fontsize=14, fontfamily='serif', pad=10)
plt.legend(frameon=True, loc='best', prop={'family': 'serif', 'size':14})
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=16, direction='in')
plt.tight_layout()

# 保存图形
plt.savefig('fig/H_z_reconstruction.pdf', bbox_inches='tight')