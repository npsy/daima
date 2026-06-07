clc; close all;

%% ================= 1. 加载所有有限角结果 =================
load('EM_Result_LimitedAngle.mat');      % 有限角 EM 结果
Result_EM = Result;

load('OSEM_Result_LimitedAngle.mat');
Result_OSEM = Result;

load('SART_Result_LimitedAngle.mat');
Result_SART = Result;

load('FBP_Result_LimitedAngle.mat', 'Result');  % 如果也跑了有限角 FBP，请确保有此文件
Result_FBP = Result;

% 真值图像（取第一个即可）
P = Result_EM.P;

% 提取 FBP 指标
psnr_fbp = Result_FBP.psnr;
ssim_fbp = Result_FBP.ssim;
img_fbp  = Result_FBP.img_fbp;

%% ================= 2. 提取收敛数据 =================
% EM
ebpn_em  = [Result_EM.metrics.EBPN];
psnr_em  = [Result_EM.metrics.PSNR];
ssim_em  = [Result_EM.metrics.SSIM];
t_em     = [Result_EM.metrics.time];

% OS-EM
ebpn_osem  = [Result_OSEM.metrics.EBPN];
psnr_osem  = [Result_OSEM.metrics.PSNR];
ssim_osem  = [Result_OSEM.metrics.SSIM];
t_osem     = [Result_OSEM.metrics.time];

% SART
ebpn_sart  = [Result_SART.metrics.EBPN];
psnr_sart  = [Result_SART.metrics.PSNR];
ssim_sart  = [Result_SART.metrics.SSIM];
t_sart     = [Result_SART.metrics.time];

%% ================= 3. PSNR vs. EBPN（保存为 PNG） =================
figure('Position',[100 100 800 500]);
hold on;
plot(ebpn_em, psnr_em, 'b-o', 'LineWidth', 2, 'DisplayName', 'EM');
plot(ebpn_osem, psnr_osem, 'g-s', 'LineWidth', 2, 'DisplayName', 'OS-EM');
plot(ebpn_sart, psnr_sart, 'r-^', 'LineWidth', 2, 'DisplayName', 'SART');
xlims = xlim;
plot(xlims, [psnr_fbp, psnr_fbp], 'k--', 'LineWidth', 1.5, 'DisplayName', 'FBP');
xlabel('Equivalent Backprojection Number');
ylabel('PSNR (dB)');
title('PSNR Convergence Comparison (Limited Angle)');
legend show; grid on;
% --- 自动保存为 PNG ---
print(gcf, 'PSNR_vs_EBPN_LimitedAngle.png', '-dpng', '-r300');

%% ================= 4. PSNR vs. Time（保存为 PNG） =================
figure('Position', [100 100 800 500]);
hold on;
plot(t_em, psnr_em, 'b-o', 'LineWidth', 2, 'DisplayName', 'EM');
plot(t_osem, psnr_osem, 'g-s', 'LineWidth', 2, 'DisplayName', 'OS-EM');
plot(t_sart, psnr_sart, 'r-^', 'LineWidth', 2, 'DisplayName', 'SART');
plot(Result_FBP.runtime, Result_FBP.psnr, 'k*', ...
     'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'FBP');
xlabel('Time (seconds)');
ylabel('PSNR (dB)');
title('PSNR vs. Running Time (Limited Angle)');
legend show; grid on;
% --- 自动保存为 PNG ---
print(gcf, 'PSNR_vs_Time_LimitedAngle.png', '-dpng', '-r300');

%% ================= 5. SSIM vs. EBPN（保存为 PNG） =================
figure('Position',[100 100 800 500]);
hold on;
plot(ebpn_em, ssim_em, 'b-o', 'LineWidth', 2, 'DisplayName', 'EM');
plot(ebpn_osem, ssim_osem, 'g-s', 'LineWidth', 2, 'DisplayName', 'OS-EM');
plot(ebpn_sart, ssim_sart, 'r-^', 'LineWidth', 2, 'DisplayName', 'SART');
xlims = xlim;
plot(xlims, [ssim_fbp, ssim_fbp], 'k--', 'LineWidth', 1.5, 'DisplayName', 'FBP');
xlabel('Equivalent Backprojection Number');
ylabel('SSIM');
title('SSIM Convergence Comparison (Limited Angle)');
legend show; grid on;
% --- 自动保存为 PNG ---
print(gcf, 'SSIM_vs_EBPN_LimitedAngle.png', '-dpng', '-r300');

%% ================= 6. SSIM vs. Time（保存为 PNG） =================
figure('Position', [100 100 800 500]);
hold on;
plot(t_em, ssim_em, 'b-o', 'DisplayName', 'EM');
plot(t_osem, ssim_osem, 'g-s', 'DisplayName', 'OS-EM');
plot(t_sart, ssim_sart, 'r-^', 'DisplayName', 'SART');
plot(Result_FBP.runtime, Result_FBP.ssim, 'k*', ...
     'MarkerSize', 12, 'DisplayName', 'FBP');
xlabel('Time (seconds)');
ylabel('SSIM');
title('SSIM vs. Running Time (Limited Angle)');
legend show; grid on;
% --- 自动保存为 PNG ---
print(gcf, 'SSIM_vs_Time_LimitedAngle.png', '-dpng', '-r300');

%% ================= 7. 最佳图像与误差图（保存为 PNG） =================
figure('Position',[100 100 1200 600]);
subplot(2,4,1); imagesc(P); axis image off; title('Ground Truth');
subplot(2,4,2); imagesc(Result_EM.best_img); axis image off; 
title(sprintf('EM Best (EBPN=%d, PSNR=%.1f dB)', Result_EM.best_iter, Result_EM.best_psnr));
subplot(2,4,3); imagesc(Result_OSEM.best_img); axis image off;
title(sprintf('OS-EM Best (EBPN=%d, PSNR=%.1f dB)', Result_OSEM.best_iter, Result_OSEM.best_psnr));
subplot(2,4,4); imagesc(Result_SART.best_img); axis image off;
title(sprintf('SART Best (EBPN=%d, PSNR=%.1f dB)', Result_SART.best_iter, Result_SART.best_psnr));

subplot(2,4,5); imagesc(abs(Result_EM.best_img - P)); axis image off; title('EM Error');
subplot(2,4,6); imagesc(abs(Result_OSEM.best_img - P)); axis image off; title('OS-EM Error');
subplot(2,4,7); imagesc(abs(Result_SART.best_img - P)); axis image off; title('SART Error');
subplot(2,4,8); imagesc(abs(img_fbp - P)); axis image off; 
title(sprintf('FBP Error (PSNR=%.1f dB)', psnr_fbp));
% --- 自动保存为 PNG ---
print(gcf, 'Best_Images_and_Errors_LimitedAngle.png', '-dpng', '-r300');

%% ================= 8. 运行时间对比（控制台输出） =================
fprintf('========== Runtime Comparison (max EBPN = 200, Limited Angle) ==========\n');
fprintf('EM    : %.2f sec\n', Result_EM.runtime);
fprintf('OS-EM : %.2f sec\n', Result_OSEM.runtime);
fprintf('SART  : %.2f sec\n', Result_SART.runtime);
fprintf('FBP   : %.2f sec\n', Result_FBP.runtime);
fprintf('==========================================================\n');

%% ================= 9. 达到特定 PSNR 所需 EBPN（控制台输出） =================
% 有限角下 PSNR 可能很低，可手动调整目标值，例如 12 dB
target_psnr = 12;   % 根据实际数据修改
if max(psnr_em) >= target_psnr
    ebpn_em_target = interp1(psnr_em, ebpn_em, target_psnr);
else
    ebpn_em_target = NaN;
end
if max(psnr_osem) >= target_psnr
    ebpn_osem_target = interp1(psnr_osem, ebpn_osem, target_psnr);
else
    ebpn_osem_target = NaN;
end
if max(psnr_sart) >= target_psnr
    ebpn_sart_target = interp1(psnr_sart, ebpn_sart, target_psnr);
else
    ebpn_sart_target = NaN;
end

fprintf('\n--- EBPN required to reach PSNR = %.1f dB (Limited Angle) ---\n', target_psnr);
fprintf('EM    : %.1f\n', ebpn_em_target);
fprintf('OS-EM : %.1f\n', ebpn_osem_target);
fprintf('SART  : %.1f\n', ebpn_sart_target);
fprintf('FBP   : 1 (fixed)\n');