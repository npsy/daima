function [image] = OSEM(x0, p, nIter, nSubset)

global SOD SDD  dNum dSize pNum pSize views

%% ================= 初始化 =================
h = pNum;

% ---------- 自适应子集数 ----------
if nargin < 4 || isempty(nSubset)
    % 自动寻找 views 的最大因子（不超过 8，且至少为 1）
    nSubset = min(8, views);
    while mod(views, nSubset) ~= 0
        nSubset = nSubset - 1;
    end
end
% 确保子集数不超过 views
nSubset = min(nSubset, views);

angle_sub = views / nSubset;            % 每个子集的角度数（必然整除）
idx = randperm(views);
angle_seq = reshape(idx, nSubset, angle_sub);  % 随机分配角度到子集

img_center = (h + 1) / 2;

if isempty(x0)
    img = ones(h, h) * 0.5;
else
    img = x0;
end

%% ================= 图像坐标 =================
u = ((1:h) - img_center) * pSize;
v = (img_center - (1:h)) * pSize;
[imgX, imgY] = meshgrid(u, v);

%% ================= 探测器坐标 =================
dh = dNum * dSize / 2;
detX = (-dh + dSize/2 : dSize : dh - dSize/2)';

%% ================= 射线模板（FOV） =================
R = SOD * dh / sqrt(dh^2 + SDD^2);
dx = pSize / 2;
intp = -R : dx : R;
singamma = detX ./ sqrt(SDD^2 + detX.^2);
cosgamma = SDD ./ sqrt(SDD^2 + detX.^2);
intX = singamma .* (intp + SOD);
intY = cosgamma .* (intp + SOD) - SOD;

%% ================= OSEM Iteration =================
for iter = 1:nIter
    fprintf('Iteration %d / %d (nSubset = %d)\n', iter, nIter, nSubset);
    for ss = 1:nSubset

        sum_back = zeros(h, h);
        sens = zeros(h, h);

        for iv = angle_seq(ss, :)

            theta = 2 * pi * (iv - 1) / views;

            c = cos(theta);
            s = sin(theta);

            rotX =  c * intX - s * intY;
            rotY =  s * intX + c * intY;

            val = interp2(imgX, imgY, img, rotX, rotY, 'linear', 0);
            proj = sum(val, 2) * dx;

            ratio = p(:, iv) ./ (proj + eps);
            ratio = min(max(ratio, 0), 10);

            rotX =  c * imgX + s * imgY;
            rotY = -s * imgX + c * imgY;

            uu = SDD .* rotX ./ (rotY + SOD);

            sum_back = sum_back + interp1(detX, ratio, uu, 'linear', 0);
            sens = sens + interp1(detX, ones(dNum, 1), uu, 'linear', 0);
        end

        img = img .* (sum_back ./ (sens + eps));
        img = max(img, 0);
    end
end
image = img;
end