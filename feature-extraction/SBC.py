import os
# import nitime
# import nitime.analysis as nta
# import nitime.fmri.io as io
# from nibabel import load
import numpy as np
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker
from nilearn.maskers import NiftiSpheresMasker
from nilearn import plotting


def run(sub, data_path, save_path="./"):
    fmri = os.path.join(data_path, f"sfnwmrda{sub}_session_1_rest_1.nii")

    pcc_coords = [(0, -52, 18)]
    mpfc_coords = [(0, 52, -6)]
    lTPJ_coords = [(-48, -54, 24)]
    rTPJ_coords = [(51, -52, 24)]

    seed_masker_pcc = NiftiSpheresMasker(
        pcc_coords, radius=8, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2, memory='nilearn_cache', memory_level=1, verbose=1)
    seed_masker_mpfc = NiftiSpheresMasker(
        mpfc_coords, radius=8, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2, memory='nilearn_cache', memory_level=1, verbose=1)
    seed_masker_lTPJ = NiftiSpheresMasker(
        lTPJ_coords, radius=8, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2, memory='nilearn_cache', memory_level=1, verbose=1)
    seed_masker_rTPJ = NiftiSpheresMasker(
        rTPJ_coords, radius=8, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2, memory='nilearn_cache', memory_level=1, verbose=1)

    seed_time_series_pcc = seed_masker_pcc.fit_transform(fmri)
    seed_time_series_mpfc = seed_masker_mpfc.fit_transform(fmri)
    seed_time_series_lTPJ = seed_masker_lTPJ.fit_transform(fmri)
    seed_time_series_rTPJ = seed_masker_rTPJ.fit_transform(fmri)

    brain_masker = NiftiMasker(
        smoothing_fwhm=6, detrend=True, standardize=True, low_pass=0.1,
        high_pass=0.01, t_r=2, memory='nilearn_cache', memory_level=1, verbose=0)

    brain_time_series = brain_masker.fit_transform(fmri)

    print("Seed time series shape(PCC): (%s, %s)" % seed_time_series_pcc.shape)
    print("Seed time series shape(MPFC): (%s, %s)" %
          seed_time_series_mpfc.shape)
    print("Seed time series shape(lTPJ): (%s, %s)" %
          seed_time_series_lTPJ.shape)
    print("Seed time series shape(rTPJ): (%s, %s)" %
          seed_time_series_rTPJ.shape)
    print("Brain time series shape: (%s, %s)" % brain_time_series.shape)

    seed_to_voxel_correlations_pcc = (
        np.dot(brain_time_series.T, seed_time_series_pcc) / seed_time_series_pcc.shape[0])
    seed_to_voxel_correlations_mpfc = (
        np.dot(brain_time_series.T, seed_time_series_mpfc) / seed_time_series_mpfc.shape[0])
    seed_to_voxel_correlations_lTPJ = (
        np.dot(brain_time_series.T, seed_time_series_lTPJ) / seed_time_series_lTPJ.shape[0])
    seed_to_voxel_correlations_rTPJ = (
        np.dot(brain_time_series.T, seed_time_series_rTPJ) / seed_time_series_rTPJ.shape[0])

    print("Seed-to-voxel correlation shape(MPFC): (%s, %s)" %
          seed_to_voxel_correlations_mpfc.shape)
    print("Seed-to-voxel correlation shape(PCC): (%s, %s)" %
          seed_to_voxel_correlations_pcc.shape)
    print("Seed-to-voxel correlation shape(lTPJ): (%s, %s)" %
          seed_to_voxel_correlations_lTPJ.shape)
    print("Seed-to-voxel correlation shape(rTPJ): (%s, %s)" %
          seed_to_voxel_correlations_rTPJ.shape)
    print("Seed-to-voxel correlation: min = %.3f; max = %.3f" % (
        seed_to_voxel_correlations_pcc.min(), seed_to_voxel_correlations_pcc.max()))
    print("Seed-to-voxel correlation: min = %.3f; max = %.3f" % (
        seed_to_voxel_correlations_mpfc.min(), seed_to_voxel_correlations_mpfc.max()))
    print("Seed-to-voxel correlation(lTPJ): min = %.3f; max = %.3f" % (
        seed_to_voxel_correlations_lTPJ.min(), seed_to_voxel_correlations_lTPJ.max()))
    print("Seed-to-voxel correlation(rTPJ): min = %.3f; max = %.3f" % (
        seed_to_voxel_correlations_rTPJ.min(), seed_to_voxel_correlations_rTPJ.max()))

    seed_to_voxel_correlations_fisher_z_pcc = np.arctanh(
        seed_to_voxel_correlations_pcc)
    print("Seed-to-voxel correlation Fisher-z transformed(pcc): min = %.3f; max = %.3f"
          % (seed_to_voxel_correlations_fisher_z_pcc.min(),
             seed_to_voxel_correlations_fisher_z_pcc.max()
             )
          )

    seed_to_voxel_correlations_fisher_z_mpfc = np.arctanh(
        seed_to_voxel_correlations_mpfc)
    print("Seed-to-voxel correlation Fisher-z transformed(mpfc): min = %.3f; max = %.3f"
          % (seed_to_voxel_correlations_fisher_z_mpfc.min(),
             seed_to_voxel_correlations_fisher_z_mpfc.max()
             )
          )

    seed_to_voxel_correlations_fisher_z_lTPJ = np.arctanh(
        seed_to_voxel_correlations_lTPJ)
    print("Seed-to-voxel correlation Fisher-z transformed(lTPJ): min = %.3f; max = %.3f"
          % (seed_to_voxel_correlations_fisher_z_lTPJ.min(),
             seed_to_voxel_correlations_fisher_z_lTPJ.max()
             )
          )

    seed_to_voxel_correlations_fisher_z_rTPJ = np.arctanh(
        seed_to_voxel_correlations_rTPJ)
    print("Seed-to-voxel correlation Fisher-z transformed(rTPJ): min = %.3f; max = %.3f"
          % (seed_to_voxel_correlations_fisher_z_rTPJ.min(),
             seed_to_voxel_correlations_fisher_z_rTPJ.max()
             )
          )

    seed_to_voxel_correlations_fisher_z_img_pcc = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z_pcc.T)
    seed_to_voxel_correlations_fisher_z_img_pcc.to_filename(
        os.path.join(save_path, 'pcc_seed_correlation_z.nii.gz'))

    seed_to_voxel_correlations_fisher_z_img_mpfc = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z_mpfc.T)
    seed_to_voxel_correlations_fisher_z_img_mpfc.to_filename(
        os.path.join(save_path, 'mpfc_seed_correlation_z.nii.gz'))

    seed_to_voxel_correlations_fisher_z_img_lTPJ = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z_lTPJ.T)
    seed_to_voxel_correlations_fisher_z_img_lTPJ.to_filename(
        os.path.join(save_path,  'lTPJ_seed_correlation_z.nii.gz'))

    seed_to_voxel_correlations_fisher_z_img_rTPJ = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z_rTPJ.T)
    seed_to_voxel_correlations_fisher_z_img_rTPJ.to_filename(
        os.path.join(save_path, 'rTPJ_seed_correlation_z.nii.gz'))


if __name__ == "__main__":
    subjects = [1018959]
    for subject in subjects:
        data_path = f"/home/arunav/Assets/ADHD200/kki_athena/KKI_preproc_filtfix/KKI/{subject}/"
        run(subject, data_path, data_path)
