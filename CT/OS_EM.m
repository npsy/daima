close all;clear;clc;
% Seting some parameters-----------------------------------------------
pNum=512;       %图像一行的像素个数，图像为M*M的大小，pixel number
dNum=512;       %探测器的单元个数，detector number
views=360;      %旋转一周的采集次数
sod=600;        %射线源焦点到旋转中心的距离
sdd=1000;       %射线源焦点到探测器中心的距离
odd=sdd-sod;    %旋转中心到探测器中心的距离
dSize=2;        %探测器单元的大小
maxIte=5;       %迭代次数
lambda=1;       %松弛因子
L=90;           %子集个数

% Some vector coordinates----------------------------------------
da=2*pi/views;    %角度间隔
dh=dNum*dSize/2;    %探测器一半的长度
R=sod*dh/sqrt((dh)^2+(sdd)^2);  %视野圆的半径
pSize=R*2/pNum; %像素的大小
dx=0.5*pSize;   %射线路径上采样点的间隔

% detector coordinates------------------------------------
detX=-dh+dSize/2:dSize:dh-dSize/2;
detX=detX';
detY=zeros(size(detX))+sdd;

% intergration points coordinates
intp=-R:dx:R; %采样范围
singamma=detX./sqrt(sdd.^2+detX.^2); %gamma为直线(连接射线源与探测器单元)与y轴的夹角
cosgamma=sdd./sqrt(sdd.^2+detX.^2);
intX = bsxfun(@times, singamma, intp+sod);  %积分点横坐标
intY = bsxfun(@times, cosgamma, intp+sod); % 积分点纵坐标(目前坐标系原点为射线源)
intY = bsxfun(@minus, intY, sod);% 做减法，变到物体坐标系，积分点纵坐标(目前坐标系原点为图像中心，世界坐标系为物体坐标系)


% Pixel coordinates------------------------------------------
temp=-R+pSize/2:pSize:R-pSize/2;
[imgX,imgY]=meshgrid(temp,temp);    %图像坐标

% Create model---------------------------------------------------
x1=0;y1=0;r1=100; %圆心为（0，0），半径为100的圆
x2=50;y2=50;r2=50; %圆心为（100，100），半径为50的圆
density1=1.0;   %密度为1
density2=2.0;   %密度为2
img=zeros(pNum,pNum);   %生成大小为512*512的图像矩阵
% 使用第一个圆的参数：x1,y1,r1,density1
img((imgX-x1).^2+(imgY-y1).^2<r1.^2)=density1;
% 使用第二个圆的参数；x2,y2,r2,density2
img((imgX-x2).^2+(imgY-y2).^2<r2.^2)=density2;
% img=phantom(512); %生成头骨幻影数据
figure;imshow(img,[]);title("Original Img")

% 划分子集-----------------------------------------------------------------
angle_sub = views/L;              %每个子集包含的角度数
angle_seq = reshape(1:views,L,angle_sub); %每个子集中包含角度

% Projection --------------------------------------------------------------
tic
proj = zeros(dNum, views);
for k = 1: views
    theta = da * k;
    rotX= cos(theta).*intX - sin(theta).*intY;
    rotY= sin(theta).*intX + cos(theta).*intY;  
    intV = interp2(imgX, imgY, img, rotX, rotY, 'linear', 0);  %积分点处值是根据像素点插值得来
    proj(:, k) = sum(intV, 2)*dx; 
end
toc
figure; imshow(proj, []), title('Projection Image');


% EM-------------------------------------------------
res=ones(pNum,pNum);
tic;
for ite=1:maxIte
    for i = 1: L
        delta=0;
        deltaImg=0;
        for k=1:angle_sub
            phi=da*angle_seq(i,k);
            rotX=cos(phi).*intX-sin(phi).*intY;
            rotY=sin(phi).*intX+cos(phi).*intY;
            intV=interp2(imgX,imgY,res,rotX,rotY,'linear',0);
            temp=sum(intV,2)*dx;
    
            for j =1:pNum
                if temp(j,1)==0
                    temp(j,1)=temp(j,1)+1e-3;
                end
            end
            delta=proj(:,angle_seq(i,k))./temp;            
            rotX=cos(phi).*imgX+sin(phi).*imgY;
            rotY=-sin(phi).*imgX+cos(phi).*imgY;
            uu=sdd.*rotX./(rotY+sod); %分母加上一个sod,变成物体坐标系
            deltaImg1=interp1(detX,delta,uu,'linear',0);  
            deltaImg=deltaImg+deltaImg1;
        end
        
        res=res.*deltaImg./angle_sub;
        
    end     
end
toc;
figure;imshow(res,[]);title('reconstruction Image');
errimg=img-res;
figure;imshow(errimg,[]);title('error Image');

