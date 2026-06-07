function [image] = SART(x0,p,nIter)

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
dx = pSize / 2; intp = -R : dx : R; 
singamma = detX ./ sqrt(SDD^2 + detX.^2); 
cosgamma = SDD ./ sqrt(SDD^2 + detX.^2); 
intX = singamma .* (intp + SOD); 
intY = cosgamma .* (intp + SOD) - SOD;
%% ================= OSEM Iteration =================
for iter = 1:nIter
    
    fprintf('Iteration %d / %d\n',iter,nIter);

    for iv = 1:views
        
        sum_back = zeros(h,h);
        sens = zeros(h,h);

        

        theta = 2*pi*(iv-1)/views;

        c = cos(theta);
        s = sin(theta);

        rotX =  c*intX - s*intY;
        rotY =  s*intX + c*intY;
            
        val = interp2(imgX, imgY, img, rotX, rotY, 'linear', 0);
        proj = sum(val,2) * dx;

        % ===============================
        % sart归一化
        % ===============================
        ones_img = ones(h,h);
        
        val_norm = interp2(imgX, imgY, ones_img, rotX, rotY, 'linear', 0);

        ray_norm = sum(val_norm,2) * dx;
            
        ratio = (p(:,iv) - proj) ./ (ray_norm + eps);

        rotX =  c*imgX + s*imgY;
        rotY =  -s*imgX + c*imgY;

        uu = SDD .* rotX ./ (rotY + SOD);

        det_ones = ones(dNum,1);

        sum_back = sum_back + interp1(detX, ratio, uu, 'linear', 0);
        sens = sens + interp1(detX, det_ones, uu, 'linear', 0);

   

        lambda = 0.02;
        correction = sum_back ./ (sens + eps);
        img = img + lambda * correction;
        img = max(img,0);
    end

end
image = img;

end