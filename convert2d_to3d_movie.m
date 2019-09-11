%%
filename = 'register.avi';
v = VideoWriter(filename);
v.FrameRate = 10;
open(v);
set(gca)
for i=1:50
    ori=imread(['E:/pyramidal/jingCode/099_00_800nm_newScanner_Part1_code_data/original/ori(',num2str(i),').png']);
    imshow(ori)
    title(num2str(i));
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)
%%
%crop layer4
y=1024;
y1=1500;
P=uint8(zeros(y1,y1-y+1,1));
%P1=uint8(zeros(4096,y1-y+1,1));
for i=1:50
    numImage=i;
    bb=imread(['E:/pyramidal/jingCode/099_00_800nm_newScanner_Part1_code_data/pred05-0.0147/099pred',num2str(i),'.png']);
    bb=bb(1:y1,y:y1);
    P(:,:,i)=bb;
end

% Jing code
CC = bwconncomp(P);
wrapper=@(x)numel(x);
C_size=cell2mat(cellfun(wrapper,CC.PixelIdxList,'un',0));
TF = isoutlier(C_size,'mean');
C_s=C_size(TF==0);
[idx,C] = kmeans(C_s',2);
min_size=min(C_s(idx==2))*2;
max_size=(max(C_size));
%max_size=max(C_s(idx==2))
for i = 1:CC.NumObjects
    if numel(CC.PixelIdxList{i})<min_size% | numel(CC.PixelIdxList{i})>max_size
        P(CC.PixelIdxList{i}) = uint8(0);
    end
end
CCC = bwconncomp(P);
wrapper=@(x)numel(x);
C_size=cell2mat(cellfun(wrapper,CCC.PixelIdxList,'un',0));
SS = regionprops(CCC,'Centroid');
centroids = cat(1, SS.Centroid);
N=length(centroids);
save('E:/pyramidal/jingCode/099_00_800nm_newScanner_Part1_code_data/center.mat','centroids')
scatter3(centroids(:,1),centroids(:,2),centroids(:,3),'.','MarkerEdgeColor','blue','MarkerFaceColor','blue')
xlabel('x'); ylabel('y'); zlabel('z');
%%
filename = 'filterd_seg.avi';
v = VideoWriter(filename);
v.FrameRate = 3;
open(v);
set(gca)
for i =1:50
    imshow(P(:,:,i))
    title(num2str(i));
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)

%%
for i=1:N
    plot3(centroids(i,1),centroids(i,2),centroids(i,3),'.','MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue')
    hold on
end
view(3);
axis tight
camlight
lighting gouraud
scalar=30;
daspect([1,1,1/scalar])
set(gca,'fontsize',18)
xlabel('x'); ylabel('y'); zlabel([num2str(scalar),'*z']);
title([num2str(N),'centers of cells with pixels ',num2str(min_size),'~',num2str(max_size)]);
box on
filename = 'test_centroid.avi';
dtheta = 1; % Must be an integer factor of 360
view(3)
v = VideoWriter(filename,'Motion JPEG AVI');
open(v);
set(gca,'CameraViewAngleMode','Manual')
for i = 1:(360/dtheta)
    camorbit(dtheta,0,'data',[0 0 1])
    drawnow
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)


%%
BW =uint8(P);%imresize(P,1/4)
clf;
[x,y,z] = meshgrid(1:size(BW,2),1:size(BW,1),1:size(BW,3));
p = patch(isosurface(x,y,z,BW,0.1));
isonormals(x,y,z,BW,p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
alpha(0.2);
hold on
for i=1:N
    plot3(centroids(i,1),centroids(i,2),centroids(i,3),'.','MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue')
    hold on
    
end
view(3);
axis tight
camlight
lighting gouraud
scalar=5;
daspect([1,1,1/scalar])

set(gca,'fontsize',18)
xlabel('x'); ylabel('y'); zlabel([num2str(scalar),'*z']);
title([num2str(N),' cells with size from ',num2str(min_size),' to ',num2str(max_size),' pixels']);
box on

filename = 'test.avi';
dtheta = 1; % Must be an integer factor of 360
view(3)
v = VideoWriter(filename,'Motion JPEG AVI');
open(v);
set(gca,'CameraViewAngleMode','Manual')
for i = 1:(360/dtheta)
    camorbit(dtheta,0,'data',[0 0 1])
    drawnow
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)


