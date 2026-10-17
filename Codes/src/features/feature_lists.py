"""Which feature table columns belong to which modality. Shared by
week6_baseline_models.py, week7_fusion_and_tuning.py, and anything later
that needs to know "what counts as optical" vs "what counts as SAR"
without redefining the list in every script.
"""

OPTICAL_FEATURES = [
    "mean_ndvi", "mean_evi", "mean_gndvi", "mean_ndwi",
    "nir_glcm_contrast", "nir_glcm_homogeneity", "nir_glcm_energy", "nir_glcm_correlation",
    "ndvi_glcm_contrast", "ndvi_glcm_homogeneity", "ndvi_glcm_energy", "ndvi_glcm_correlation",
]

SAR_FEATURES = [
    "mean_vv_db", "std_vv_db", "mean_vh_db", "std_vh_db", "mean_vv_minus_vh_db",
    "vv_glcm_contrast", "vv_glcm_homogeneity", "vv_glcm_energy", "vv_glcm_correlation",
    "vh_glcm_contrast", "vh_glcm_homogeneity", "vh_glcm_energy", "vh_glcm_correlation",
]
