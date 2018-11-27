close all
clear all
%------------------------------------
%Getting name of folders in image dir.
images_dir='Frames';
folder_list=dir(images_dir);
num_folder=size(folder_list,1);

%Cinverting image into k*k image, factor is used to calculate boxsize for
%creating patch of image.winsize is the parameter used in LPQ feature 
k=60;
factor=2;
boxsize=k/factor;
winsize=3;
%Iterating over folders. First two entries are not folder.
for i=3:num_folder
    if(folder_list(i).isdir)
        folderpath=strcat(folder_list(i).folder,'\',folder_list(i).name);
        files_list=dir(folderpath);
        num_files=size(files_list,1);
        %Iterating over files. First two entries are not files.
        for j=3:num_files
            if(files_list(j).isdir==0)
                filepath=strcat(files_list(j).folder,'\',files_list(i).name);
                filename=strsplit(files_list(j).name,'.');
                filename=string(filename(1));
                txtname=strcat(filename,'.txt');
                txtfilepath=strcat(files_list(i).folder,'\',txtname);
                %Image Pre-Processing - Grayscale conversion and recaling
                % to k*k
                img_rgb=imread(filepath);
                img_gray=rgb2gray(img_rgb);
                img=imresize(img_gray,[k,k]);
                %LPQ feature extraction
                LPQ_feature=[];
                for p=1:boxsize:k
                    for q=1:boxsize:k
                        patch=img(p:p+boxsize-1,q:q+boxsize-1);
                        LPQ_hist=lpq(patch,winsize);
                        LPQ_feature=[LPQ_feature,LPQ_hist];
                    end
                end
                fileID = fopen(txtfilepath,'w');
                fprintf(fileID,'%f,',LPQ_feature);
                fclose(fileID);
            end
        end
    
    end
end
% %Image Pre-Processing
% img_rgb=imread('test.jpg');
% img_gray=rgb2gray(img_rgb);
% %Dataset resolution k*k
% k=60;
% factor=2;
% boxsize=k/factor;
% img=imresize(img_gray,[k,k]);
% figure()
% imshow(img)
% title('Image Processed')
% %LPQ feature extraction
% LPQ_feature=[];
% for i=1:boxsize:k
%     for j=1:boxsize:k
%         patch=img(i:i+boxsize-1,j:j+boxsize-1);
%         LPQ_hist=lpq(patch,3);
%         LPQ_feature=[LPQ_feature,LPQ_hist];
%     end
% end
