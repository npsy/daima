clc;
close all;

%% ============================================================
% Geometry (same as your SART script)
%% ============================================================
global SOD SDD ODD dNum dSize pNum pSize views

SOD   = 1000;
SDD   = 1500;
ODD   = SDD - SOD;

dNum  = 256;
dSize = 1.0;

pNum  = 256;
pSize = 1.0;

views = 180;          % 初始全采样角度数

%% ============================================================
% Phantom (identical to SART script)
%% ============================================================
dh = dNum * dSize / 2;
R_fov = SOD * dh / sqrt(dh^2 + SDD^2);   % mm

h = pNum;
img_center = (h + 1) / 2;
u = ((1:h) - img_center) * pSize;
v = (img_center - (1:h)) * pSize;
[imgX, imgY] = meshgrid(u, v);

P = phantom('Modified Shepp-Logan', h);
P = max(P, 0);
P = P ./ max(P(:));

margin = 1.0;   % 1 mm safety margin
mask = (imgX.^2 + imgY.^2) <= (R_fov - margin)^2;
P = P .* mask;
P = max(P, 0);
if max(P(:)) > 0
    P = P ./ max(P(:));
end

figure;
imagesc(u, v, P);
axis image off;
colormap gray;
colorbar;
title('Ground Truth (FOV-cropped)');

%% ============================================================
% Projection (fan-beam)
%% ============================================================
fprintf('Generating fan-beam projection...\n');
proj_fan = forward_project_cone(P);

proj_fan = real(proj_fan);
proj_fan(isnan(proj_fan)) = eps;
proj_fan(proj_fan <= 0) = eps;

% ========== 有限角修改：只保留 0°~120° 的投影 ==========
angle_vec = (0:views-1) * 360/views;      % 原始角度向量（0°,2°,4°,...）
limit_angle = 120;                        % 有限角范围（度）
idx_keep = angle_vec <= limit_angle;      % 逻辑索引
proj_fan = proj_fan(:, idx_keep);         % 截取正弦图
views = sum(idx_keep);                    % 更新全局变量 views
% ============================================================

figure;
imagesc((0:views-1)*360/views, 1:dNum, proj_fan);
axis tight;
colormap gray;
colorbar;
xlabel('View (deg)');
ylabel('Detector');
title('Fan-beam Sinogram (Limited Angle)');

%% ============================================================
% FBP Reconstruction (fan-beam -> rebin -> parallel FBP)
%% ============================================================
fprintf('\nStarting FBP reconstruction...\n');

tic;

% Step 1: Rebinning to parallel beam
[P_para, theta_para, s_para] = rebin_fan2para(proj_fan);

% Step 2: Ramp filtering
d_s = s_para(2) - s_para(1);
P_filt = filter_sinogram(P_para, d_s);   % Ram-Lak filter, full bandwidth

% Step 3: Backprojection
img_fbp = fbp_backproject(P_filt, theta_para, s_para);

runtime = toc;

% Post-processing
img_fbp = real(img_fbp);
img_fbp(isnan(img_fbp)) = 0;
img_fbp(isinf(img_fbp)) = 0;
img_fbp = max(img_fbp, 0);

%% ============================================================
% Metrics
%% ============================================================
mse_val = mean((img_fbp(:) - P(:)).^2);
psnr_val = psnr(img_fbp, P);
ssim_val = ssim(img_fbp, P);

fprintf('FBP Results:\n');
fprintf('  MSE  = %.6f\n', mse_val);
fprintf('  PSNR = %.3f dB\n', psnr_val);
fprintf('  SSIM = %.5f\n', ssim_val);
fprintf('  Runtime = %.3f sec\n', runtime);

%% ============================================================
% Display Results
%% ============================================================
figure('Position', [100 100 1200 400]);
subplot(1,3,1)
imagesc(P);
axis image off;
colormap gray;
colorbar;
title('Ground Truth');

subplot(1,3,2)
imagesc(img_fbp);
axis image off;
colormap gray;
colorbar;
title('FBP Reconstruction (Limited Angle)');

subplot(1,3,3)
imagesc(abs(img_fbp - P));
axis image off;
colormap hot;
colorbar;
title('Error Map');

%% ============================================================
% Save Results
%% ============================================================
Result.views     = views;
Result.psnr      = psnr_val;
Result.ssim      = ssim_val;
Result.mse       = mse_val;
Result.runtime   = runtime;
Result.img_fbp   = img_fbp;

save('FBP_Result_LimitedAngle.mat', 'Result');

fprintf('\nFinished.\n');