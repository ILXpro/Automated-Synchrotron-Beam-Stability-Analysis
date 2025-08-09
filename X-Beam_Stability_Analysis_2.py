import numpy as np
from PIL import Image
from scipy import ndimage, optimize, interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure, morphology, exposure
import os
import csv
import re
from matplotlib.ticker import ScalarFormatter

def natural_key(string_):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', string_)]

class EnhancedSpotAnalyzer:
    def __init__(self, image_path, pixel_size_um=0.325):
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.original_image = self.load_image_raw(image_path)
        self.pixel_size_um = pixel_size_um
        self.center_x = None
        self.center_y = None
        self.center_err_x = None
        self.center_err_y = None
        self.angle_rad = None
        self.angle_deg = None
        self.angle_err_deg = None
        self.profile_length = 300
        self.profile_width = 10
        self.ortho_length = 100
        self.ortho_width = 10
        self.spot_mask = None
        self.spot_region = None
        self.original_min = None
        self.original_max = None
        self.bit_depth = None
        self.true_bit_depth = None

    def load_image_raw(self, path):
        with Image.open(path) as img:
            original_mode = img.mode
            if original_mode == 'I;16':
                img_32 = img.convert('I')
                img_array = np.array(img_32, dtype=np.float64)
                max_value = img_array.max()
                self.true_bit_depth = 8 if max_value <= 255 else 16 if max_value <= 65535 else 32
                gray_array = img_array
            elif original_mode == 'I':
                img_array = np.array(img, dtype=np.float64)
                max_value = img_array.max()
                self.true_bit_depth = 8 if max_value <= 255 else 16 if max_value <= 65535 else 32
                gray_array = img_array
            elif original_mode == 'RGB':
                img_array = np.array(img, dtype=np.float64)
                gray_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
                self.true_bit_depth = 8
            elif original_mode in ['L', 'P']:
                img_array = np.array(img, dtype=np.float64)
                gray_array = img_array
                self.true_bit_depth = 8
            else:
                img_converted = img.convert('L')
                gray_array = np.array(img_converted, dtype=np.float64)
                self.true_bit_depth = 8
            self.original_min = gray_array.min()
            self.original_max = gray_array.max()
            self.bit_depth = self.true_bit_depth
            return gray_array

    def detect_spot_region(self):
        from skimage.filters import threshold_local
        block_size = min(self.original_image.shape) // 10 * 2 + 1
        if block_size % 2 == 0:
            block_size += 1
        adaptive_thresh = threshold_local(self.original_image, block_size=block_size, method='gaussian')
        mask = self.original_image > adaptive_thresh
        cleaned = morphology.remove_small_objects(mask, min_size=50)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
        labeled = measure.label(cleaned)
        regions = measure.regionprops(labeled, intensity_image=self.original_image)
        if not regions:
            raise ValueError("Не удалось обнаружить пятно на изображении")
        brightest_region = max(regions, key=lambda r: r.mean_intensity)
        self.spot_mask = labeled == brightest_region.label
        self.spot_region = brightest_region
        return self.spot_mask

    def calculate_orientation_and_center_error(self):
        if self.spot_mask is None:
            self.detect_spot_region()
        y_coords, x_coords = np.where(self.spot_mask)
        weights = self.original_image[y_coords, x_coords]
        mean_x = np.average(x_coords, weights=weights)
        mean_y = np.average(y_coords, weights=weights)
        std_x = np.sqrt(np.average((x_coords - mean_x)**2, weights=weights))
        std_y = np.sqrt(np.average((y_coords - mean_y)**2, weights=weights))
        n_eff = np.sum(weights)**2 / np.sum(weights**2)
        self.center_x = mean_x
        self.center_y = mean_y
        self.center_err_x = std_x / np.sqrt(n_eff)
        self.center_err_y = std_y / np.sqrt(n_eff)
        coords = np.column_stack((x_coords, y_coords))
        centered = coords - [mean_x, mean_y]
        cov = np.cov(centered.T, aweights=weights)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_vector = eigvecs[:, np.argmax(eigvals)]
        angle_rad = np.arctan2(principal_vector[1], principal_vector[0])
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        angle_rad = np.radians(angle_deg)
        sigma_major = np.sqrt(np.max(eigvals))
        sigma_minor = np.sqrt(np.min(eigvals))
        angle_err_rad = 0.5 * (sigma_minor / sigma_major) / np.sqrt(n_eff)
        self.angle_rad = angle_rad
        self.angle_deg = angle_deg
        self.angle_err_deg = np.degrees(angle_err_rad)
        return mean_x, mean_y, self.center_err_x, self.center_err_y, angle_deg, self.angle_err_deg

    def extract_intensity_profile(self, direction='main'):
        if direction == 'main':
            length = self.profile_length
            width = self.profile_width
            dx = np.cos(self.angle_rad)
            dy = np.sin(self.angle_rad)
        else:
            length = self.ortho_length
            width = self.ortho_width
            dx = -np.sin(self.angle_rad)
            dy = np.cos(self.angle_rad)
        perp_dx = -dy
        perp_dy = dx
        t_values = np.linspace(-length/2, length/2, length)
        profile_intensities = []
        for t in t_values:
            main_x = self.center_x + t * dx
            main_y = self.center_y + t * dy
            width_intensities = []
            for w in np.linspace(-width/2, width/2, width):
                sample_x = main_x + w * perp_dx
                sample_y = main_y + w * perp_dy
                if (0 <= sample_x < self.original_image.shape[1] and
                    0 <= sample_y < self.original_image.shape[0]):
                    intensity = ndimage.map_coordinates(
                        self.original_image, [[sample_y], [sample_x]],
                        order=1, mode='nearest', prefilter=False
                    )[0]
                    width_intensities.append(intensity)
            if width_intensities:
                profile_intensities.append(np.mean(width_intensities))
        return np.array(profile_intensities)

    def calculate_fwhm_direct(self, profile):
        if len(profile) < 5:
            return 0.0, 0.0, None
        max_intensity = np.max(profile)
        min_intensity = np.min(profile)
        if max_intensity == min_intensity:
            return 0.0, 0.0, None
        half_max = min_intensity + (max_intensity - min_intensity) / 2
        x_data = np.arange(len(profile))
        interp_func = interpolate.interp1d(x_data, profile, kind='linear', bounds_error=False, fill_value=min_intensity)
        x_fine = np.linspace(0, len(profile)-1, len(profile)*10)
        profile_fine = interp_func(x_fine)
        above_half = profile_fine >= half_max
        crossings = np.where(np.diff(above_half.astype(int)))[0]
        if len(crossings) < 2:
            return 0.0, 0.0, None
        left_idx = crossings[0]
        right_idx = crossings[-1]
        left_pos = x_fine[left_idx] + (half_max - profile_fine[left_idx]) * (
            x_fine[left_idx+1] - x_fine[left_idx]) / (profile_fine[left_idx+1] - profile_fine[left_idx])
        right_pos = x_fine[right_idx] + (half_max - profile_fine[right_idx]) * (
            x_fine[right_idx+1] - x_fine[right_idx]) / (profile_fine[right_idx+1] - profile_fine[right_idx])
        fwhm = right_pos - left_pos
        grad = np.gradient(profile_fine, x_fine)
        grad_left = np.abs(grad[left_idx])
        grad_right = np.abs(grad[right_idx])
        window = 10
        left_noise = np.std(profile_fine[max(0, left_idx-window):left_idx+window])
        right_noise = np.std(profile_fine[max(0, right_idx-window):right_idx+window])
        left_err = left_noise / grad_left if grad_left > 1e-6 else 0
        right_err = right_noise / grad_right if grad_right > 1e-6 else 0
        fwhm_error = np.sqrt(left_err**2 + right_err**2)
        return fwhm, fwhm_error, (left_pos, right_pos)

    def calculate_fwhm_gauss(self, profile):
        if len(profile) < 5:
            return 0.0, 0.0, None
        x = np.arange(len(profile))
        p0 = [np.max(profile)-np.min(profile), np.argmax(profile), len(profile)/5, np.min(profile)]
        try:
            popt, pcov = optimize.curve_fit(
                lambda x, a, x0, sigma, offset: offset + a * np.exp(-0.5 * ((x - x0) / sigma) ** 2),
                x, profile, p0=p0)
            sigma = np.abs(popt[2])
            if np.isfinite(pcov[2,2]) and pcov[2,2] > 0:
                sigma_err = np.sqrt(np.abs(pcov[2,2]))
            else:
                sigma_err = sigma * 0.1
            fwhm = 2.355 * sigma
            fwhm_err = 2.355 * sigma_err
            fit_curve = popt[3] + popt[0] * np.exp(-0.5 * ((x - popt[1]) / popt[2]) ** 2)
            return fwhm, fwhm_err, fit_curve
        except Exception:
            return 0.0, 0.0, None

    def evaluate_profile_criterion(self, profile_length, fwhm, fwhm_error, axis='major'):
        messages = []
        if self.spot_region:
            axis_length = self.spot_region.major_axis_length if axis == 'major' else self.spot_region.minor_axis_length
            ratio = profile_length / axis_length if axis_length > 0 else 0
            if ratio < 1.0:
                messages.append("Длина профиля < размера пятна — FWHM занижен!")
            elif ratio < 1.5:
                messages.append("Длина профиля чуть больше пятна. Рекомендуется увеличить.")
            elif ratio > 3.0:
                messages.append("Длина профиля слишком большая — возможно искажение FWHM.")
            else:
                messages.append("Длина профиля оптимальна.")
            if fwhm_error is not None and fwhm is not None and fwhm > 0:
                rel_error = fwhm_error / fwhm
                if rel_error > 0.3:
                    messages.append(f"Высокая ошибка FWHM ({rel_error:.1%})!")
                elif rel_error > 0.15:
                    messages.append(f"Ошибка FWHM ({rel_error:.1%}) на границе. Увеличьте ширину профиля.")
                else:
                    messages.append(f"Ошибка FWHM ({rel_error:.1%}) в норме.")
            else:
                messages.append("Не удалось оценить ошибку FWHM.")
        else:
            messages.append("Область пятна не определена.")
        return messages

    def get_safe_display_range(self, roi, bit_depth=16, max_saturation=0.01, gamma=None, use_log=False):
        roi = roi.astype(np.float64)
        vmin = roi.min()
        vmax_raw = roi.max()
        saturation_level = (2**bit_depth - 1)
        sat_pixels = np.sum(roi >= saturation_level)
        sat_ratio = sat_pixels / roi.size
        if sat_ratio > max_saturation:
            vmax = np.percentile(roi, 99.9)
        else:
            vmax = vmax_raw
        vmax = vmax * 1.05
        roi_disp = roi.copy()
        if use_log:
            roi_disp = np.log1p(roi_disp - vmin)
        if gamma is not None:
            roi_disp = exposure.adjust_gamma(roi_disp, gamma)
        return vmin, vmax, roi_disp, sat_ratio

    def save_technical_figure(self, main_profile, ortho_profile,
                             main_fwhm, main_fwhm_error, main_fwhm_gauss, main_fwhm_gauss_err, main_fit_curve,
                             ortho_fwhm, ortho_fwhm_error, ortho_fwhm_gauss, ortho_fwhm_gauss_err, ortho_fit_curve,
                             main_criteria, ortho_criteria, output_dir,
                             use_log=False, gamma=None):
        pixel_size_um = self.pixel_size_um
        max_len = max(self.profile_length, self.ortho_length)
        roi_size_um = 1.2 * max_len * pixel_size_um
        roi_xc_um = self.center_x * pixel_size_um
        roi_yc_um = self.center_y * pixel_size_um
        roi_xmin_um = roi_xc_um - roi_size_um / 2
        roi_xmax_um = roi_xc_um + roi_size_um / 2
        roi_ymin_um = roi_yc_um - roi_size_um / 2
        roi_ymax_um = roi_yc_um + roi_size_um / 2
        roi_xmin = max(0, int(roi_xmin_um / pixel_size_um))
        roi_xmax = min(self.original_image.shape[1], int(roi_xmax_um / pixel_size_um))
        roi_ymin = max(0, int(roi_ymin_um / pixel_size_um))
        roi_ymax = min(self.original_image.shape[0], int(roi_ymax_um / pixel_size_um))
        roi = self.original_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax].copy()
        bit_depth = self.bit_depth if self.bit_depth else 16
        vmin, vmax, roi_disp, sat_ratio = self.get_safe_display_range(
            roi, bit_depth=bit_depth, max_saturation=0.01, gamma=gamma, use_log=use_log
        )
        fig = plt.figure(figsize=(13, 8), constrained_layout=False)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.18], width_ratios=[1.1, 1], wspace=0.45, hspace=0.6)
        ax_img = fig.add_subplot(gs[0:2, 0])
        extent = [roi_xmin * pixel_size_um, roi_xmax * pixel_size_um, roi_ymax * pixel_size_um, roi_ymin * pixel_size_um]
        im = ax_img.imshow(roi_disp, cmap='gray', origin='upper', extent=extent,
                           interpolation='none', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax_img)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Интенсивность', rotation=270, labelpad=15)
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.update_ticks()
        center_marker = Circle((self.center_x*pixel_size_um, self.center_y*pixel_size_um),
                               radius=0.5*pixel_size_um, color='#FFD600', fill=True, zorder=3)
        ax_img.add_patch(center_marker)
        for color, length, width, dx, dy, label in [
            ('#1976D2', self.profile_length, self.profile_width, np.cos(self.angle_rad), np.sin(self.angle_rad), 'Основной профиль'),
            ('#F57C00', self.ortho_length, self.ortho_width, -np.sin(self.angle_rad), np.cos(self.angle_rad), 'Ортогональный профиль')
        ]:
            perp_dx = -dy
            perp_dy = dx
            rect = Rectangle(
                ((self.center_x - dx*length/2 - perp_dx*width/2)*pixel_size_um,
                 (self.center_y - dy*length/2 - perp_dy*width/2)*pixel_size_um),
                length*pixel_size_um, width*pixel_size_um,
                angle=np.degrees(np.arctan2(dy, dx)),
                linewidth=1.3, edgecolor=color, facecolor='none', alpha=0.8, zorder=2, label=label
            )
            ax_img.add_patch(rect)
        legend = ax_img.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='gray', fancybox=True)
        legend.get_frame().set_alpha(0.8)
        scalebar = ScaleBar(1, 'um', fixed_value=20, fixed_units='um',
                            location='lower right', color='white', box_alpha=0.7, scale_loc='bottom')
        ax_img.add_artist(scalebar)
        ax_img.set_title(
            f"Угол ориентации: {self.angle_deg:.2f}° ± {self.angle_err_deg:.2f}°\n"
            f"Центр: ({self.center_x*pixel_size_um:.1f} ± {self.center_err_x*pixel_size_um:.1f}, "
            f"{self.center_y*pixel_size_um:.1f} ± {self.center_err_y*pixel_size_um:.1f}) мкм",
            fontsize=11, pad=14)
        ax_img.set_xlabel("X, мкм", fontsize=11, labelpad=12)
        ax_img.set_ylabel("Y, мкм", fontsize=11, labelpad=12)
        ax_img.tick_params(labelsize=10)
        ax_main = fig.add_subplot(gs[0, 1])
        main_x_centered = (np.arange(len(main_profile)) - len(main_profile)//2) * pixel_size_um
        ax_main.scatter(main_x_centered, main_profile, color='#1976D2', s=14, label='Основной профиль')
        if main_fit_curve is not None:
            ax_main.plot(main_x_centered, main_fit_curve, 'k--', lw=1.0, label='Гаусс-фит')
        ax_main.set_title("Профиль вдоль главной оси", fontsize=11, color='#1976D2', pad=10)
        ax_main.set_xlabel("Положение, мкм", fontsize=11, labelpad=12)
        ax_main.set_ylabel("Интенсивность, отн. ед.", fontsize=11, labelpad=12)
        ax_main.tick_params(labelsize=10)
        ax_main.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                       frameon=True, facecolor='white', edgecolor='gray', fancybox=True)
        ax_main.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_main.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax_ortho = fig.add_subplot(gs[1, 1])
        ortho_x_centered = (np.arange(len(ortho_profile)) - len(ortho_profile)//2) * pixel_size_um
        ax_ortho.scatter(ortho_x_centered, ortho_profile, color='#F57C00', s=14, label='Ортогональный профиль')
        if ortho_fit_curve is not None:
            ax_ortho.plot(ortho_x_centered, ortho_fit_curve, 'k-.', lw=1.0, label='Гаусс-фит')
        ax_ortho.set_title("Профиль вдоль ортогональной оси", fontsize=11, color='#F57C00', pad=10)
        ax_ortho.set_xlabel("Положение, мкм", fontsize=11, labelpad=12)
        ax_ortho.set_ylabel("Интенсивность, отн. ед.", fontsize=11, labelpad=12)
        ax_ortho.tick_params(labelsize=10)
        ax_ortho.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                        frameon=True, facecolor='white', edgecolor='gray', fancybox=True)
        ax_ortho.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_ortho.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax_caption = fig.add_subplot(gs[2, :])
        ax_caption.axis('off')
        ax_caption.text(
            0.5, 0.5,
            f"FWHM осн.: {main_fwhm*pixel_size_um:.3f} ± {main_fwhm_error*pixel_size_um:.3f} мкм | "
            f"Гаусс: {main_fwhm_gauss*pixel_size_um:.3f} ± {main_fwhm_gauss_err*pixel_size_um:.3f} мкм\n"
            f"FWHM орт.: {ortho_fwhm*pixel_size_um:.3f} ± {ortho_fwhm_error*pixel_size_um:.3f} мкм | "
            f"Гаусс: {ortho_fwhm_gauss*pixel_size_um:.3f} ± {ortho_fwhm_gauss_err*pixel_size_um:.3f} мкм\n"
            f"Критерии осн.: {'; '.join(main_criteria)}\n"
            f"Критерии орт.: {'; '.join(ortho_criteria)}\n"
            f"Автодиапазон: {vmin:.0f} – {vmax:.0f}\n"
            f"Доля насыщенных пикселей: {sat_ratio*100:.2f}%",
            ha='center', va='center', fontsize=9, color='black', wrap=True, transform=ax_caption.transAxes
        )
        plt.savefig(os.path.join(output_dir, f"{self.image_name}_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_publication_figure(self, main_profile, ortho_profile,
                               main_fwhm, main_fwhm_error, main_fwhm_gauss, main_fwhm_gauss_err, main_fit_curve,
                               ortho_fwhm, ortho_fwhm_error, ortho_fwhm_gauss, ortho_fwhm_gauss_err, ortho_fit_curve,
                               output_dir,
                               use_log=False, gamma=None):
        pixel_size_um = self.pixel_size_um
        max_len = max(self.profile_length, self.ortho_length)
        roi_size_um = 1.2 * max_len * pixel_size_um
        roi_xc_um = self.center_x * pixel_size_um
        roi_yc_um = self.center_y * pixel_size_um
        roi_xmin_um = roi_xc_um - roi_size_um / 2
        roi_xmax_um = roi_xc_um + roi_size_um / 2
        roi_ymin_um = roi_yc_um - roi_size_um / 2
        roi_ymax_um = roi_yc_um + roi_size_um / 2
        roi_xmin = max(0, int(roi_xmin_um / pixel_size_um))
        roi_xmax = min(self.original_image.shape[1], int(roi_xmax_um / pixel_size_um))
        roi_ymin = max(0, int(roi_ymin_um / pixel_size_um))
        roi_ymax = min(self.original_image.shape[0], int(roi_ymax_um / pixel_size_um))
        roi = self.original_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax].copy()
        bit_depth = self.bit_depth if self.bit_depth else 16
        vmin, vmax, roi_disp, sat_ratio = self.get_safe_display_range(
            roi, bit_depth=bit_depth, max_saturation=0.01, gamma=gamma, use_log=use_log
        )
        fig = plt.figure(figsize=(13, 6), constrained_layout=False)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.18], width_ratios=[1.1, 1], wspace=0.45, hspace=0.5)
        ax_img = fig.add_subplot(gs[0, 0])
        extent = [roi_xmin * pixel_size_um, roi_xmax * pixel_size_um, roi_ymax * pixel_size_um, roi_ymin * pixel_size_um]
        ax_img.imshow(roi_disp, cmap='gray', origin='upper', extent=extent,
                      interpolation='none', vmin=vmin, vmax=vmax)
        scalebar = ScaleBar(1, 'um', fixed_value=20, fixed_units='um',
                            location='lower right', color='white', box_alpha=0.7, scale_loc='bottom')
        ax_img.add_artist(scalebar)
        ax_img.set_xlabel("X, мкм", fontsize=11, labelpad=12)
        ax_img.set_ylabel("Y, мкм", fontsize=11, labelpad=12)
        ax_img.set_title("ROI фокального пятна", fontsize=12, pad=10)
        ax_img.tick_params(labelsize=10)
        ax_prof = fig.add_subplot(gs[0, 1])
        main_x_centered = (np.arange(len(main_profile)) - len(main_profile)//2) * pixel_size_um
        ortho_x_centered = (np.arange(len(ortho_profile)) - len(ortho_profile)//2) * pixel_size_um
        ax_prof.scatter(main_x_centered, main_profile, s=16, color='#1976D2', label='Основной')
        if main_fit_curve is not None:
            ax_prof.plot(main_x_centered, main_fit_curve, 'k--', lw=1.0, label='Гаусс-фит осн.')
        ax_prof.scatter(ortho_x_centered, ortho_profile, s=16, color='#F57C00', label='Ортогональный')
        if ortho_fit_curve is not None:
            ax_prof.plot(ortho_x_centered, ortho_fit_curve, 'k-.', lw=1.0, label='Гаусс-фит орт.')
        ax_prof.set_xlabel("Положение, мкм", fontsize=11, labelpad=12)
        ax_prof.set_ylabel("Интенсивность, отн. ед.", fontsize=11, labelpad=12)
        ax_prof.tick_params(labelsize=10)
        ax_prof.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                       frameon=True, facecolor='white', edgecolor='gray', fancybox=True)
        ax_prof.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_prof.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax_caption = fig.add_subplot(gs[1, :])
        ax_caption.axis('off')
        ax_caption.text(
            0.5, 0.5,
            f"FWHM осн.: {main_fwhm*pixel_size_um:.2f} ± {main_fwhm_error*pixel_size_um:.2f} мкм | "
            f"Гаусс: {main_fwhm_gauss*pixel_size_um:.2f} ± {main_fwhm_gauss_err*pixel_size_um:.2f} мкм\n"
            f"FWHM орт.: {ortho_fwhm*pixel_size_um:.2f} ± {ortho_fwhm_error*pixel_size_um:.2f} мкм | "
            f"Гаусс: {ortho_fwhm_gauss*pixel_size_um:.2f} ± {ortho_fwhm_gauss_err*pixel_size_um:.2f} мкм\n"
            f"Доля насыщенных пикселей: {sat_ratio*100:.2f}%",
            ha='center', va='center', fontsize=9, color='black', wrap=True, transform=ax_caption.transAxes
        )
        plt.savefig(os.path.join(output_dir, f"{self.image_name}_publication.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

def process_single_file(image_path, params, output_dir):
    try:
        analyzer = EnhancedSpotAnalyzer(image_path, pixel_size_um=params['pixel_size_um'])
        analyzer.detect_spot_region()
        analyzer.calculate_orientation_and_center_error()
        analyzer.profile_length = params['main_profile_length']
        analyzer.profile_width = params['main_profile_width']
        analyzer.ortho_length = params['ortho_profile_length']
        analyzer.ortho_width = params['ortho_profile_width']

        main_profile = analyzer.extract_intensity_profile(direction='main')
        ortho_profile = analyzer.extract_intensity_profile(direction='ortho')

        main_fwhm, main_fwhm_error, _ = analyzer.calculate_fwhm_direct(main_profile)
        ortho_fwhm, ortho_fwhm_error, _ = analyzer.calculate_fwhm_direct(ortho_profile)

        main_fwhm_gauss, main_fwhm_gauss_err, main_fit_curve = analyzer.calculate_fwhm_gauss(main_profile)
        ortho_fwhm_gauss, ortho_fwhm_gauss_err, ortho_fit_curve = analyzer.calculate_fwhm_gauss(ortho_profile)

        main_criteria = analyzer.evaluate_profile_criterion(analyzer.profile_length, main_fwhm, main_fwhm_error, axis='major')
        ortho_criteria = analyzer.evaluate_profile_criterion(analyzer.ortho_length, ortho_fwhm, ortho_fwhm_error, axis='minor')

        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        analyzer.save_technical_figure(
            main_profile, ortho_profile,
            main_fwhm, main_fwhm_error, main_fwhm_gauss, main_fwhm_gauss_err, main_fit_curve,
            ortho_fwhm, ortho_fwhm_error, ortho_fwhm_gauss, ortho_fwhm_gauss_err, ortho_fit_curve,
            main_criteria, ortho_criteria, images_dir
        )
        analyzer.save_publication_figure(
            main_profile, ortho_profile,
            main_fwhm, main_fwhm_error, main_fwhm_gauss, main_fwhm_gauss_err, main_fit_curve,
            ortho_fwhm, ortho_fwhm_error, ortho_fwhm_gauss, ortho_fwhm_gauss_err, ortho_fit_curve,
            images_dir
        )

        results_dict = {
            "file": os.path.basename(image_path),
            "center_x_um": round(analyzer.center_x * params['pixel_size_um'], 3),
            "center_x_um_err": round(analyzer.center_err_x * params['pixel_size_um'], 3),
            "center_y_um": round(analyzer.center_y * params['pixel_size_um'], 3),
            "center_y_um_err": round(analyzer.center_err_y * params['pixel_size_um'], 3),
            "angle_deg": round(analyzer.angle_deg, 3),
            "angle_deg_err": round(analyzer.angle_err_deg, 3),
            "main_profile_length_um": round(analyzer.profile_length * params['pixel_size_um'], 3),
            "main_profile_width_um": round(analyzer.profile_width * params['pixel_size_um'], 3),
            "ortho_profile_length_um": round(analyzer.ortho_length * params['pixel_size_um'], 3),
            "ortho_profile_width_um": round(analyzer.ortho_width * params['pixel_size_um'], 3),
            "main_fwhm_um": round(main_fwhm * params['pixel_size_um'], 3),
            "main_fwhm_um_err": round(main_fwhm_error * params['pixel_size_um'], 3),
            "main_fwhm_gauss_um": round(main_fwhm_gauss * params['pixel_size_um'], 3),
            "main_fwhm_gauss_um_err": round(main_fwhm_gauss_err * params['pixel_size_um'], 3),
            "ortho_fwhm_um": round(ortho_fwhm * params['pixel_size_um'], 3),
            "ortho_fwhm_um_err": round(ortho_fwhm_error * params['pixel_size_um'], 3),
            "ortho_fwhm_gauss_um": round(ortho_fwhm_gauss * params['pixel_size_um'], 3),
            "ortho_fwhm_gauss_um_err": round(ortho_fwhm_gauss_err * params['pixel_size_um'], 3),
            "main_criteria": '; '.join(main_criteria),
            "ortho_criteria": '; '.join(ortho_criteria)
        }
        return results_dict

    except Exception as e:
        print(f"[ERROR] Детальная ошибка для {os.path.basename(image_path)}: {str(e)}")
        return None

def process_folder(folder_path, params):
    output_dir = os.path.join(folder_path, 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    files = sorted(files, key=natural_key)
    for file in files:
        image_path = os.path.join(folder_path, file)
        result = process_single_file(image_path, params, output_dir)
        if result is not None:
            all_results.append(result)
        print(f"[INFO] Обработан файл: {file}")

    if all_results:
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        summary_csv_path = os.path.join(output_dir, f"{folder_name}_analysis_results.csv")
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[INFO] Сводная таблица сохранена: {summary_csv_path}")
        print(f"[INFO] Все изображения сохранены в: {os.path.join(output_dir, 'images')}")
        print(f"[INFO] Обработано файлов: {len(all_results)}")

if __name__ == "__main__":
    input_path = "Data/0.001s/"
    params = {
        'pixel_size_um': 0.325,
        'main_profile_length': 350,
        'main_profile_width': 5,
        'ortho_profile_length': 37,
        'ortho_profile_width': 10
    }

    if os.path.isdir(input_path):
        process_folder(input_path, params)
    elif os.path.isfile(input_path):
        output_dir = os.path.join(os.path.dirname(input_path), 'analysis_output')
        os.makedirs(output_dir, exist_ok=True)
        result = process_single_file(input_path, params, output_dir)
        if result is not None:
            csv_path = os.path.join(output_dir, 'results.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writeheader()
                writer.writerow(result)
            print(f"[INFO] Результаты сохранены в: {output_dir}")
    else:
        print("Ошибка: указанный путь не существует.")
