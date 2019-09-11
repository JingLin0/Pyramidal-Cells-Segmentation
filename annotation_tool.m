%%
%350:2397,350:2397
numImage=50;
if ~exist(['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\original\\ori(',num2str(numImage),').png'],'file')
    data=imread(['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\099_00_800nm_newScanner_Part1\\ori(',num2str(numImage),').png']);
    data=data(1024:2047,1024:2047);
    imwrite(data,['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\original\\ori(',num2str(numImage),').png'])
else
    data=imread(['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\original\\ori(',num2str(numImage),').png']);
end


if ~exist(['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\unrefined_binary\\099ori',num2str(numImage),'a.png'], 'file')
bb=zeros(size(data,1),size(data,2));
imwrite(bb,['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\unrefined_binary\\099ori',num2str(numImage),'a.png']);
else
    bb=imread(['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\unrefined_Binary\\099ori',num2str(numImage),'a.png']);
    if max(bb)==255
    bb=bb/255;
    end
end
bb=logical(bb);

%%
 %draw edge of cells and fill it
figure
imshowpair(data,bb)
nnn=50;
for i=1:10
    
    
[p1,p2] = ginput(1);
p1=ceil(p1);
p2=ceil(p2);
if p1<2 || p1>size(data,1)-5|| p2<2 || p2>size(data,1)-5
    continue
else
    break
   
end
end
%bb=bi_data(:,:,numImage);
ZOOM=true;
if ZOOM

xlim(p1+[-nnn,nnn])
ylim(p2+[-nnn,nnn])
sz=4;
for i=1:900
[a1,b1]=ginput(1);
a1=round(a1);
b1=round(b1);
if bb(b1,a1)==1
   % bb(b1,a1)=0;
    bb(b1:(b1+sz),a1:(a1+sz))=0;
%bb(b1-sz:b1+sz,a1-sz:a1+sz)=zeros(sz*2+1);
else
    bb(b1:(b1+sz),a1:(a1+sz))=1;
end
aa=imfill(bb,'holes');
bb=aa(1:1024,1:1024);
imshowpair(data,bb);

xlim(p1+[-nnn,nnn])
ylim(p2+[-nnn,nnn])
imwrite(bb,['E:\\pyramidal\\jingCode\\099_00_800nm_newScanner_Part1_code_data\\unrefined_binary\\099ori',num2str(numImage),'a.png']);
end

end