clc;clear all;
% path='E:\\Align test_save\\';
folder ='D:\\pyramidalData\\099_00_800nm_newScanner_Part1\\';
fileName = '1';
srcFiles = dir([folder,'/*.tif']);  % the folder in which ur images exists
path=srcFiles.folder;
path=[path '\'];
ANNOTATE = false;
ZOOM = true;
VISUALIZE = true;
SAVEIMAGE = true;
% Rough alignment registration
% Maybe change the parameters to increase the speed
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.01;
original = uint8(rgb2gray(imread([path,fileName,' (1).tif'])));
%% Rough Alignment
h = gca;
% Do the Radon transform, så the picture is horizontal
theta = 0:180;
[R0,xp0] = radon(original,theta);
% Find the location of the peak of the radon transform image.
maxR0 = max(R0(:));
[rowOfMax0, columnOfMax0] = find(R0 == maxR0);
J = imrotate(original, -columnOfMax0+90); % Rotated image!!! if we regard the first image is good
verHigh = 3050; %define if the length of the columns is above 3050, since it is unnatual
% Read files in folder
for i = 2:size(srcFiles,1)
    disp(['Register image number ',num2str(i),' out of ',num2str(length(srcFiles))])
    fixed = original;%J(:,:,i-1);
    moving = uint8(rgb2gray(imread([path,fileName',' (',num2str(i),').tif'])));
    % Rotate if the vertical axis is much bigger than horienzontal
    if size(moving,1)>verHigh && abs(diff([size(moving,1),size(moving,2)]))>round(min(size(moving))*0.45)==1
        bw = im2bw(moving);
        bw = bwareaopen(bw, 200000); % remove small white areas
        bwfill= imfill(~bw,'holes');
        R=regionprops(bwareafilt(~bwfill,1000),'Orientation'); % Find the orientation
        oriMean=mean([R.Orientation]); % average orientation
        rotDeg=oriMean+90;
        % result=(x-90*sign(x))
        result =oriMean-rotDeg; % we want to rotate it 90 degree when it is vertical
        moving = imrotate(moving,result,'bilinear');
        [R,~] = radon(moving,theta);
        % Find the location of the peak of the radon transform image.
        maxR = max(R(:));
        [rowOfMax, columnOfMax] = find(R == maxR);
        if columnOfMax<70==1
            moving_rodon = imrotate(moving, -columnOfMax); % Rotated image!!!
        elseif columnOfMax>90 && columnOfMax<150==1
            moving_rodon = imrotate(moving, -columnOfMax+105); % Rotated image!!!
        else
            moving_rodon = imrotate(moving, -columnOfMax+90); % Rotated image!!!
        end
        moving_reg = imregister(moving_rodon,fixed,'rigid',optimizer,metric);
        J(:,:,i)= moving_reg;
        if VISUALIZE
            imshow(uint8(moving_reg),'Parent',h); drawnow;
        end
    else
        [R,xp] = radon(moving,theta);
        % Find the location of the peak of the radon transform image.
        maxR = max(R(:));
        [rowOfMax, columnOfMax] = find(R == maxR);
        rotDiff =-columnOfMax+90;
        if rotDiff<0 && rotDiff>-5 &&rowOfMax<1100 && rowOfMax>980==1
            rotDiff =-columnOfMax+80; % keep the same difference
        elseif rotDiff>20&& rotDiff<40&&rowOfMax>2500==1
            % if the difference is above 10clc
            rotDiff =-10;
        elseif rotDiff<40 && rotDiff>35==1
            % if the difference is above 10
            rotDiff =-columnOfMax+40; % keep the same difference
        elseif rotDiff>10 && rotDiff>-5&& rotDiff>40==1
            % if the difference is above 10clc
            rotDiff =-columnOfMax; % keep the same difference
        elseif rotDiff<-5 && rotDiff>-10==1
            %              rotDiff =-columnMax(j)+86;
            rotDiff =-10;
        elseif rotDiff<-10==1
            rotDiff =12;
        else
        end
        moving_rodon = moving;%imrotate(moving, rotDiff); % Rotated image!!!
        
        moving_reg = imregister(moving_rodon,fixed,'rigid',optimizer,metric);
        J(:,:,i)= moving_reg;
        if VISUALIZE
            imshow(uint8(moving_reg),'Parent',h); title(i);drawnow;
        end
    end
end
%%

folder = 'alignedImg_J_50';
fileName ='Img_rough';
folderName=[folder];
figure 
imshowpair(original,moving_reg)
%imwrite(moving_reg,[folderName,'/',fileName,'_0',num2str(2),'.tif'],'png');


%% Save the first align images to make an comparison later
if SAVEIMAGE
    folder = 'alignedImg_J_50';
    fileName ='Img_rough';
    folderName=[folder];
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
    for i = 1:size(J,3)
        %             conImg = mat2gray(L(:,:,i)); % Convert to a gray image!
        imwrite(J(:,:,i),[folderName,'/',fileName,'_0',num2str(i),'.tif'],'png');
    end
end
save('J_48(good).mat','J')
%% Automatic refine landmark registration on a subset of the image stack
% Define if it should be the whole image or crop image
CROP = false;
if CROP
    %     Automatic crop image
    %     corner = [1000,800,1];
    %     x=J(corner(1)+[0:1024],corner(2)+[0:1024],corner(3)+[0:size(J,3)-1]);
    % Manuel Crop the image
    [I1c cRect] = imcrop(imshow(J(:,:,20)));
    cRect = round(cRect);
    x = zeros(cRect(4)+1,cRect(3)+1,size(J,3),'uint8');
    for i = 1:size(J,3)
        x(:,:,i) = imcrop(J(:,:,i),cRect);
    end
    disp('Crop done')
else
    x = J;% Align the whole image and not just a subset
end
K = double(x); %subimage double
%% Apply intensity-based image registration

K=J;
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.001; % choose a small radius

h = gca;
tform = cell(size(K,3),1);
tform{1} = affine2d; % Identity matrix
M = tform{1}.T;
L = K;
for i = 2:size(K,3)
    disp(['Process image number ',num2str(i),' out of ',num2str(size(K,3))])
    t = imregtform(K(:,:,i), K(:,:,i-1), 'rigid', optimizer, metric); %Align fixed and moving image
    tform{i} = t;
    M = t.T*M;
    % Apply the transformation
    L(:,:,i) = imwarp(K(:,:,i),affine2d(M),'OutputView',imref2d([size(K,1),size(K,2)]));
    if VISUALIZE
        imagesc(L(:,:,i),'Parent',h); colormap(gray); axis image; title(i); drawnow;
    end
end

%%
load D:/MATLAB/R2018b/bin/alignedImg_J_50/L_align_48.mat
%%
%%%%%%%%%%%%%%%%%%%%%%%Save aligned image %%%%%%%%%%%%%%%%%%%%%%
SAVEIMAGE=1
if SAVEIMAGE
    %     folder = 'aligned_final_48';
    folder = 'alignedImg_align_final_50';
    fileName ='ori';
    folderName=[folder];
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
    for i = 1:size(L,3)
        conImg = mat2gray(L(:,:,i)); % Convert to a gray image!
        %         conImg = mat2gray(x(:,:,i)); % Convert to a gray image!
        imwrite(conImg, [folderName,'/',fileName,'(',num2str(i),').png'],'png');
    end
    save('K_align_48.mat','K')
    %save('L_align_48.mat','L')
    L8bit = uint8(255 * mat2gray(L));
    save('L8bit_align_48.mat','L8bit')
end
