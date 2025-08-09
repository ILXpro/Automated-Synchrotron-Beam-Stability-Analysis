import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- Физические ограничения ---
PIXEL_SIZE_UM = 0.325  # размер пикселя, мкм
PSF_UM = 0.5           # PSF детектора, мкм (пример, заменить на ваш)
SIGMA_DET = max(PIXEL_SIZE_UM / 2, PSF_UM / 2.355)  # PSF переводим в σ, если задана FWHM

# --- Настройки шрифтов ---
BIG_FONT = 24
plt.rcParams.update({
    'font.size': BIG_FONT,
    'axes.titlesize': BIG_FONT,
    'axes.labelsize': BIG_FONT,
    'xtick.labelsize': BIG_FONT,
    'ytick.labelsize': BIG_FONT,
    'legend.fontsize': BIG_FONT,
    'figure.titlesize': BIG_FONT,
})

def extract_shot_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else np.nan

def calc_total_error(fit_err, sigma_det=SIGMA_DET):
    return np.sqrt(np.square(fit_err) + sigma_det**2)

def plot_with_uncertainty(ax, x, y, yerr, label, color, title, ylabel):
    y_smooth = gaussian_filter1d(y, sigma=2)
    yerr_smooth = gaussian_filter1d(yerr, sigma=2)
    ax.plot(x, y_smooth, color=color, linewidth=2, label=label)
    ax.fill_between(x, y_smooth - yerr_smooth, y_smooth + yerr_smooth, color=color, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel('Номер снимка')
    ax.set_ylabel(ylabel)
    #ax.legend()
    ax.grid(False)

def plot_with_points(ax, x, y, yerr, label, color, title, ylabel):
    ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, label=label, capsize=6)
    ax.set_title(title)
    ax.set_xlabel('Номер снимка')
    ax.set_ylabel(ylabel)
    #ax.legend()
    ax.grid(False)

def analyze_spot_evolution(csv_path):
    if not os.path.isfile(csv_path):
        print(f"Файл {csv_path} не найден!")
        return

    output_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)
    df['shot_number'] = df['file'].apply(extract_shot_number)
    df = df.sort_values('shot_number').reset_index(drop=True)

    # Центрирование X и Y
    x_mean = df['center_x_um'].mean()
    y_mean = df['center_y_um'].mean()
    df['center_x_um_centered'] = df['center_x_um'] - x_mean
    df['center_y_um_centered'] = df['center_y_um'] - y_mean

    shots = df['shot_number']

    # --- Итоговые погрешности ---
    error_pairs = [
        ('center_x_um_centered', 'center_x_um_err'),
        ('center_y_um_centered', 'center_y_um_err'),
        ('ortho_fwhm_um', 'ortho_fwhm_um_err'),
        ('ortho_fwhm_gauss_um', 'ortho_fwhm_gauss_um_err'),
        ('main_fwhm_um', 'main_fwhm_um_err'),
        ('main_fwhm_gauss_um', 'main_fwhm_gauss_um_err'),
    ]
    for val_col, err_col in error_pairs:
        if val_col in df.columns and err_col in df.columns:
            df[f'{err_col}_total'] = calc_total_error(df[err_col], SIGMA_DET)

    # --- Визуализация: 3 строки × 2 колонки ---
    fig, axs = plt.subplots(3, 2, figsize=(18, 18), constrained_layout=True)
    # Левая колонка: ΔY, FWHM орт. (Gauss), FWHM орт. (прямое)
    plot_with_uncertainty(
        axs[0, 0], shots, df['center_y_um_centered'], df['center_y_um_err_total'],
        label='ΔY', color='#F57C00', title='Y относительно среднего', ylabel='ΔY, мкм'
    )
    if 'ortho_fwhm_gauss_um' in df.columns and 'ortho_fwhm_gauss_um_err_total' in df.columns:
        plot_with_uncertainty(
            axs[1, 0], shots, df['ortho_fwhm_gauss_um'], df['ortho_fwhm_gauss_um_err_total'],
            label='FWHM орт. (Gauss)', color='#FFB300',
            title='FWHM орт. (Gauss)', ylabel='FWHM, мкм'
        )
    if 'ortho_fwhm_um' in df.columns and 'ortho_fwhm_um_err_total' in df.columns:
        plot_with_uncertainty(
            axs[2, 0], shots, df['ortho_fwhm_um'], df['ortho_fwhm_um_err_total'],
            label='FWHM орт. (прямое)', color='#D32F2F',
            title='FWHM орт. (прямое)', ylabel='FWHM, мкм'
        )
    # Правая колонка: ΔX, FWHM осн. (Gauss), FWHM осн. (прямое)
    plot_with_uncertainty(
        axs[0, 1], shots, df['center_x_um_centered'], df['center_x_um_err_total'],
        label='ΔX', color='#1976D2', title='X относительно среднего', ylabel='ΔX, мкм'
    )
    if 'main_fwhm_gauss_um' in df.columns and 'main_fwhm_gauss_um_err_total' in df.columns:
        plot_with_uncertainty(
            axs[1, 1], shots, df['main_fwhm_gauss_um'], df['main_fwhm_gauss_um_err_total'],
            label='FWHM осн. (Gauss)', color='#00BFAE',
            title='FWHM осн. (Gauss)', ylabel='FWHM, мкм'
        )
    if 'main_fwhm_um' in df.columns and 'main_fwhm_um_err_total' in df.columns:
        plot_with_uncertainty(
            axs[2, 1], shots, df['main_fwhm_um'], df['main_fwhm_um_err_total'],
            label='FWHM осн. (прямое)', color='#388E3C',
            title='FWHM осн. (прямое)', ylabel='FWHM, мкм'
        )
    plt.suptitle('Эволюция положения и размера фокального пятна (относительно среднего)', fontsize=BIG_FONT+4)
    fig_path = os.path.join(output_dir, f'{base_name}_spot_evolution_centered.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # --- Классический стиль: точки с погрешностями ---
    fig2, axs2 = plt.subplots(3, 2, figsize=(18, 18), constrained_layout=True)
    plot_with_points(
        axs2[0, 0], shots, df['center_y_um_centered'], df['center_y_um_err_total'],
        label='ΔY', color='#F57C00', title='Y относительно среднего', ylabel='ΔY, мкм'
    )
    if 'ortho_fwhm_gauss_um' in df.columns and 'ortho_fwhm_gauss_um_err_total' in df.columns:
        plot_with_points(
            axs2[1, 0], shots, df['ortho_fwhm_gauss_um'], df['ortho_fwhm_gauss_um_err_total'],
            label='FWHM орт. (Gauss)', color='#FFB300',
            title='FWHM орт. (Gauss)', ylabel='FWHM, мкм'
        )
    if 'ortho_fwhm_um' in df.columns and 'ortho_fwhm_um_err_total' in df.columns:
        plot_with_points(
            axs2[2, 0], shots, df['ortho_fwhm_um'], df['ortho_fwhm_um_err_total'],
            label='FWHM орт. (прямое)', color='#D32F2F',
            title='FWHM орт. (прямое)', ylabel='FWHM, мкм'
        )
    plot_with_points(
        axs2[0, 1], shots, df['center_x_um_centered'], df['center_x_um_err_total'],
        label='ΔX', color='#1976D2', title='X относительно среднего', ylabel='ΔX, мкм'
    )
    if 'main_fwhm_gauss_um' in df.columns and 'main_fwhm_gauss_um_err_total' in df.columns:
        plot_with_points(
            axs2[1, 1], shots, df['main_fwhm_gauss_um'], df['main_fwhm_gauss_um_err_total'],
            label='FWHM осн. (Gauss)', color='#00BFAE',
            title='FWHM осн. (Gauss)', ylabel='FWHM, мкм'
        )
    if 'main_fwhm_um' in df.columns and 'main_fwhm_um_err_total' in df.columns:
        plot_with_points(
            axs2[2, 1], shots, df['main_fwhm_um'], df['main_fwhm_um_err_total'],
            label='FWHM осн. (прямое)', color='#388E3C',
            title='FWHM осн. (прямое)', ylabel='FWHM, мкм'
        )
    fig2_path = os.path.join(output_dir, f'{base_name}_spot_evolution_points.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # --- Итоговая статистика и тренды ---
    stats = {}
    for col, name in [
        ('center_x_um_centered', 'ΔX, мкм'),
        ('center_y_um_centered', 'ΔY, мкм'),
        ('ortho_fwhm_um', 'FWHM орт. (прямое)'),
        ('ortho_fwhm_gauss_um', 'FWHM орт. (Gauss)'),
        ('main_fwhm_um', 'FWHM осн. (прямое)'),
        ('main_fwhm_gauss_um', 'FWHM осн. (Gauss)')
    ]:
        if col in df.columns:
            stats[name] = (df[col].mean(), df[col].std(), df[col].min(), df[col].max())
    stats_table = pd.DataFrame(stats, index=['Среднее', 'Ст. откл.', 'Мин.', 'Макс.']).T

    trends = {}
    for col, name in [
        ('center_x_um_centered', 'ΔX, мкм'),
        ('center_y_um_centered', 'ΔY, мкм'),
        ('ortho_fwhm_um', 'FWHM орт. (прямое)'),
        ('ortho_fwhm_gauss_um', 'FWHM орт. (Gauss)'),
        ('main_fwhm_um', 'FWHM осн. (прямое)'),
        ('main_fwhm_gauss_um', 'FWHM осн. (Gauss)')
    ]:
        if col in df.columns:
            fit = np.polyfit(shots, df[col], 1)
            trends[name] = fit[0]

    # --- Поиск файлов с FWHM орт. (Gauss), наиболее близких к среднему ---
    col_fwhm = 'ortho_fwhm_gauss_um'
    closest_files = None
    if col_fwhm in df.columns:
        mean_fwhm = df[col_fwhm].mean()
        df['fwhm_diff'] = np.abs(df[col_fwhm] - mean_fwhm)
        closest_idx = df['fwhm_diff'].nsmallest(3).index
        closest_files = df.loc[closest_idx, ['file', col_fwhm, 'fwhm_diff']].copy()
        closest_files = closest_files.rename(columns={
            'file': 'Имя файла',
            col_fwhm: 'FWHM орт. (Gauss), мкм',
            'fwhm_diff': 'Отклонение от среднего, мкм'
        })
        print('\nФайлы с FWHM орт. (Gauss), наиболее близкими к среднему:')
        print(closest_files.to_string(index=False))

    # --- Запись в итоговый CSV ---
    out_csv = os.path.join(output_dir, f'{base_name}_stats_centered.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    with open(out_csv, 'a', encoding='utf-8-sig') as f:
        f.write('\n# Итоговая статистика (относительно среднего)\n')
        stats_table.to_csv(f, float_format='%.4f')
        f.write('\n# Тренды (мкм/шаг)\n')
        for k, v in trends.items():
            f.write(f'{k}: {v:.5f}\n')
        f.write(f'\n# Погрешность детектора (использована для свертки): {SIGMA_DET:.3f} мкм\n')
        f.write('# Итоговые погрешности рассчитаны как sqrt(σ_det^2 + σ_fit^2)\n')
        f.write('\n# Файлы с FWHM орт. (Gauss), наиболее близкими к среднему\n')
        if closest_files is not None:
            closest_files.to_csv(f, index=False, float_format='%.4f')
        else:
            f.write('Нет данных по ortho_fwhm_gauss_um\n')

    print(f'Погрешность детектора (σ_det): {SIGMA_DET:.3f} мкм')
    print('\nИтоговая статистика по параметрам (относительно среднего):')
    print(stats_table)
    print('\nТренды (мкм/шаг):')
    for k, v in trends.items():
        print(f'{k}: {v:.5f}')
    print(f'\nГрафики и полный CSV сохранены в {output_dir}')

# Пример запуска:
#analyze_spot_evolution('Data/0.01s/analysis_output/0.01s_analysis_results.csv')
analyze_spot_evolution('Data/0.2s/analysis_output/0.2s_analysis_results.csv')