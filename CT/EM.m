function [image] = EM(x0,p,nIter)

global SOD SDD ODD dNum dSize pNum pSize views

%% ================= 初始化 =================
h = pNum;
img_center = (h + 1) / 2;

if isempty(x0)
    img = ones(h,h) * 0.5;
else
    img = x0;
end

%% ================= 图像坐标 =================
u = ((1:h) - img_center) * pSize;
v = (img_center - (1:h)) * pSize;
[imgX,imgY] = meshgrid(u,v);

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

%% ================= EM Iteration =================
for iter = 1:nIter
    
    fprintf('Iteration %d / %d\n',iter,nIter);

    sum_back = zeros(h,h);
    sens     = zeros(h,h);
    
    for iv = 1:views
        
        theta = 2*pi*(iv-1)/views;

        c = cos(theta);
        s = sin(theta);

        %% =====================================
        % Forward （统一旋转方向）
        %% =====================================
        rotX =  c*intX - s*intY;
        rotY =  s*intX + c*intY;

        val = interp2(imgX, imgY, img, rotX, rotY, 'linear', 0);

        proj = sum(val,2) * dx;

        ratio = p(:,iv) ./ (proj + eps);
        ratio = min(max(ratio,0),10);

        %% =====================================
        % Backward（❗关键修改在这里）
        %% =====================================

        % 使用 SAME rotation（和forward一致）
        rotX =  c*imgX + s*imgY;
        rotY =  -s*imgX + c*imgY;

        % 扇束坐标映射（保持一致）
        uu = SDD .* rotX ./ (rotY + SOD);

        back = interp1(detX, ratio, uu, 'linear', 0);
        sum_back = sum_back + back;

        % sensitivity
        tmp = interp1(detX, ones(dNum,1), uu, 'linear', 0);
        sens = sens + tmp;
        
    end
        
    %% =====================================
    % EM Update
    %% =====================================
    img = img .* (sum_back ./ (sens + eps));

    img(isnan(img)) = 0;
    img(isinf(img)) = 0;
    img = max(img,0);

    fprintf('   max = %.4f   min = %.4f\n',max(img(:)),min(img(:)));

end

image = img;

end