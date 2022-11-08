% Runs the preprocessing pipeline
% Authors:  Arunav Dey, Jigya Singh, Manaswini Rathore, Roshni Govind

spm_jobman('initcfg');

subs = [1018959 1019436 1043241 1266183 1535233 1541812 1577042 1594156 1623716 1638334 1652369 1686265];

rootdir = '/home/arunav/Assets/ADHD200/kki_niak/anat_kki';
% rootdir = '/home/arunav/Assets/Capstone/Oxford';

% spmdir = '/home/arunav/matlab/spm12';
spmdir = '/mnt/c/Users/aruna/OneDrive/Documents/MATLAB/spm12';

for sub = subs
    sub = num2str(sub, '%05d');

    if exist([rootdir 'X_' sub '/func/rest.nii']) == 0
        gunzip([rootdir '/sub' sub '/func/rest.nii.gz']);
    else
    end

    if exist([rootdir '/sub' sub '/anat/mprage_anonymized.nii']) == 0
        gunzip([rootdir '/sub' sub '/anat/mprage_anonymized.nii.gz']);
    else
    end

    % Selecting the .nii file
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.cfg_named_file.name = 'restfmri';
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.cfg_named_file.files = {{[rootdir '/sub' sub '/func/rest.nii']}};

    % Realignment
    matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1) = cfg_dep('Named File Selector: restfmri(1) - Files', substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'files', '{}', {1}));
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.weight = '';
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.which = [2 1];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.mask = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.prefix = 'r';

    % Slice time correction
    matlabbatch{3}.spm.temporal.st.scans{1}(1) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 1)', substruct('.', 'val', '{}', {2}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'sess', '()', {1}, '.', 'rfiles'));
    matlabbatch{3}.spm.temporal.st.nslices = 29;
    matlabbatch{3}.spm.temporal.st.tr = 2;
    matlabbatch{3}.spm.temporal.st.ta = 1.93103448275862;
    matlabbatch{3}.spm.temporal.st.so = [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 2 4 6 8 10 12 14 16 18 20 22 24 26 28];
    matlabbatch{3}.spm.temporal.st.refslice = 1;
    matlabbatch{3}.spm.temporal.st.prefix = 'a';

    % Coregistration
    matlabbatch{4}.spm.spatial.coreg.estwrite.ref(1) = cfg_dep('Realign: Estimate & Reslice: Mean Image', substruct('.', 'val', '{}', {2}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'rmean'));
    matlabbatch{4}.spm.spatial.coreg.estwrite.source = {[rootdir '/sub' sub '/anat/mprage_anonymized.nii,1']};
    matlabbatch{4}.spm.spatial.coreg.estwrite.other = {''};
    matlabbatch{4}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
    matlabbatch{4}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
    matlabbatch{4}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{4}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
    matlabbatch{4}.spm.spatial.coreg.estwrite.roptions.interp = 4;
    matlabbatch{4}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{4}.spm.spatial.coreg.estwrite.roptions.mask = 0;
    matlabbatch{4}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

    % Tissue segmentation
    matlabbatch{5}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate & Reslice: Coregistered Images', substruct('.', 'val', '{}', {4}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'cfiles'));
    matlabbatch{5}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{5}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{5}.spm.spatial.preproc.channel.write = [0 1];
    matlabbatch{5}.spm.spatial.preproc.tissue(1).tpm = {[spmdir '/tpm/TPM.nii,1']};
    matlabbatch{5}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{5}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(2).tpm = {[spmdir '/tpm/TPM.nii,2']};
    matlabbatch{5}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{5}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(3).tpm = {[spmdir '/tpm/TPM.nii,3']};
    matlabbatch{5}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{5}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(4).tpm = {[spmdir '/tpm/TPM.nii,4']};
    matlabbatch{5}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{5}.spm.spatial.preproc.tissue(4).native = [1 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(5).tpm = {[spmdir '/tpm/TPM.nii,5']};
    matlabbatch{5}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{5}.spm.spatial.preproc.tissue(5).native = [1 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(6).tpm = {[spmdir '/tpm/TPM.nii,6']};
    matlabbatch{5}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{5}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{5}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{5}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{5}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{5}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{5}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{5}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{5}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{5}.spm.spatial.preproc.warp.write = [0 1];
    matlabbatch{5}.spm.spatial.preproc.warp.vox = NaN;
    matlabbatch{5}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                            NaN NaN NaN];

    % Normalisation
    matlabbatch{6}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.', 'val', '{}', {5}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'fordef', '()', {':'}));
    matlabbatch{6}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', substruct('.', 'val', '{}', {3}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('()', {1}, '.', 'files'));
    matlabbatch{6}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                        78 76 85];
    matlabbatch{6}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{6}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{6}.spm.spatial.normalise.write.woptions.prefix = 'w';

    % Smoothing
    matlabbatch{7}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.', 'val', '{}', {6}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('()', {1}, '.', 'files'));
    matlabbatch{7}.spm.spatial.smooth.fwhm = [6 6 6];
    matlabbatch{7}.spm.spatial.smooth.dtype = 0;
    matlabbatch{7}.spm.spatial.smooth.im = 0;
    matlabbatch{7}.spm.spatial.smooth.prefix = 's';

    % Output files
    matlabbatch{8}.cfg_basicio.file_dir.file_ops.cfg_file_split.name = 'preproc';
    matlabbatch{8}.cfg_basicio.file_dir.file_ops.cfg_file_split.files(1) = cfg_dep('Smooth: Smoothed Images', substruct('.', 'val', '{}', {7}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), substruct('.', 'files'));
    matlabbatch{8}.cfg_basicio.file_dir.file_ops.cfg_file_split.index = {
                                                                    1
                                                                    2
                                                                    }';
    spm_jobman('run', matlabbatch);

end
