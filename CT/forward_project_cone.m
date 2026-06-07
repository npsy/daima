function proj = forward_project_cone(image)
% =====================================================
% Fan-beam Forward Projection (Template-based Version)
%  EM forward projector
% =====================================================

global SOD SDD dNum dSize pNum pSize views

h = pNum;
img_center = (h+1)/2;

%% ---------- Detector coordinates ----------
dh = dNum*dSize/2;
detX = (-dh+dSize/2 : dSize : dh-dSize/2)';   % detector channel center

%% ---------- FOV ----------
Robj = SOD * dh / sqrt(dh^2 + SDD^2);

%% ---------- Sampling step ----------
ds = pSize/2;

%% ---------- Build 0-degree ray template ----------
intp = -Robj : ds : Robj;

singamma = detX ./ sqrt(SDD^2 + detX.^2);
cosgamma = SDD ./ sqrt(SDD^2 + detX.^2);

rayX0 = singamma .* (intp + SOD);
rayY0 = cosgamma .* (intp + SOD) - SOD;

% rayX0 / rayY0 : dNum × Ns
Ns = length(intp);

%% ---------- Image coordinate ----------
u = ((1:h)-img_center)*pSize;
[X,Y] = meshgrid(u,u);

proj = zeros(dNum,views);

%% =====================================================
% Loop views
%% =====================================================
for iv = 1:views
    
    theta = 2*pi*(iv-1)/views;

    c = cos(theta);
    s = sin(theta);

    %% rotate ray template
    rotX = c*rayX0 - s*rayY0;
    rotY = s*rayX0 + c*rayY0;

    %% world -> image index
    col = rotX/pSize + img_center;
    row = img_center - rotY/pSize;

    %% interpolate
    vals = interp2(image,col,row,'linear',0);

    %% line integral
    proj(:,iv) = sum(vals,2)*ds;
end

proj(proj<=0) = eps;

end