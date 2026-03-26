import cv2
import numpy as np
import os

INPUT_DIR = 'dataset/Spina_Bifida_Real'
OUTPUT_DIR = 'dataset/Spina_Bifida_Real_Split'

PANEL_LAYOUTS = {
    'pmc8436766_fig2_chiari_lemon_banana.jpg': (1, 3),
    'pmc8436766_fig3_ventriculomegaly_cranial.jpg': (1, 3),
    'pmc4945507_fig1_lemon_sign_spine.jpg': (1, 2),
    'pmc4945507_fig2_banana_sign_chiari.jpg': (1, 2),
    'pmc4945507_fig5_spinal_dysraphism.jpg': (1, 2),
    'pmc4945507_fig6_spinal_tether.jpg': (1, 2),
    'pmc7700296_fig1_normal_anatomy.jpg': (1, 3),
    'pmc7700296_fig2_case1_brainstem.jpg': (1, 2),
    'pmc7700296_fig3_case2_lemon_banana.jpg': (2, 2),
    'pmc7700296_fig4_case3_myelomeningocele.jpg': (2, 2),
    'pmc7700296_fig5_case4_ntd.jpg': (2, 2),
    'pmc7700296_fig6_case5_absent_IT.jpg': (1, 2),
    'pmc7700296_fig7_case6_encephalocele.jpg': (1, 2),
    'pmc4898641_fig4_mmc_lemon_banana_signs.jpg': (2, 4),
    'pmc9857041_fig3_osb_case1.jpg': (2, 3),
    'pmc9857041_fig4_osb_case3.jpg': (2, 3),
    'pmc9857041_fig5_cephalocele.jpg': (2, 3),
    'pmc7545505_fig1_spinal_defect_brain.jpg': (3, 3),
    'pmc7545505_fig2_body_stalk.jpg': (2, 3),
    'pmc11269449_fig1_lumbosacral_ntd.jpg': (1, 2),
    'pmc5024920_fig2_normal_3d_spine.jpg': (1, 3),
    'pmc5024920_fig3_transverse_spine.jpg': (2, 2),
    'pmc5024920_fig7_multiplanar_conus.jpg': (1, 3),
    'pmc5024920_fig10_abnormal_curvature.jpg': (1, 3),
    'pmc5024920_fig11_cleft_vertebrae.jpg': (1, 2),
    'pmc5024920_fig12_block_vertebrae.jpg': (1, 2),
    'pmc5024920_fig13_hemivertebra.jpg': (1, 2),
    'pmc5024920_fig14_chiari_ii_cranial.jpg': (2, 2),
    'pmc5024920_fig15_open_ntd_vertebral.jpg': (1, 3),
    'pmc5024920_fig16_3d_open_ntd.jpg': (1, 2),
    'pmc5024920_fig17_lipomeningocele.jpg': (1, 3),
    'pmc5024920_fig18_lipomeningocele_mixed.jpg': (1, 2),
    'pmc5024920_fig20_sacral_agenesis.jpg': (1, 2),
    'pmc5024920_fig21_sacral_agenesis_sag.jpg': (1, 2),
}


def split_image_grid(img, rows, cols):
    h, w = img.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    panels = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            panel = img[y1:y2, x1:x2]
            if np.mean(panel) < 250:
                panels.append(panel)
    return panels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_panels = 0

    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"  Skipping: {filename}")
            continue

        base = os.path.splitext(filename)[0]

        if filename in PANEL_LAYOUTS:
            rows, cols = PANEL_LAYOUTS[filename]
            panels = split_image_grid(img, rows, cols)
            for i, panel in enumerate(panels):
                out_path = os.path.join(OUTPUT_DIR, f"{base}_panel{i+1}.jpg")
                cv2.imwrite(out_path, panel)
                total_panels += 1
            print(f"  {filename} -> {len(panels)} panels")
        else:
            out_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(out_path, img)
            total_panels += 1
            print(f"  {filename} (single)")

    print(f"\nTotal images produced: {total_panels}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
