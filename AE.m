clc;
clear all;
No_of_classes = 40;
Images_per_class = 10;
Training_images = 6;
T_Train_Images =zeros(40,240);
T_Test_Images = zeros(40,160);
Train_cell = {};
Test_cell = {};
TrainImages = {};
TestImages = {};

for i = 1:No_of_classes
    fn = cd(['C:\Users\Ritika\Desktop\Neural Networks\Ritika_Chowdri_AE\s' num2str(i)]);
    for j = 1:Training_images
        filename = [num2str(j) '.pgm'];
        img =imread(filename);
        img = imresize(img,0.5);
        img = double(img)/256;
        Train_cell = mat2cell(img,56,46);
        TrainImages = [Train_cell TrainImages];
        
    end
    
    for k = 7:Images_per_class
        filename = [num2str(j) '.pgm'];
        img =imread(filename);
        img = imresize(img,0.5);
        img = double(img)/256;
        Test_cell = mat2cell(img,56,46);
        TestImages = [Test_cell TestImages];
    end
end

for i= 1:40
    for j=1:6:240
        T_Train_Images(i,j:j+5) = ones();
        i = i+1;
    end
    break;
end

for i= 1:40
    for j=1:4:160
        T_Test_Images(i,j:j+3) = ones();
        i = i+1;
    end
    break;
end

%% Autoencoder
rng('default');
Hidden_size1 = 100;

for L2WR = 0.006:0.002:0.008
    for SR = 1:2:3
        for SP = 0.15:0.15:0.30
            autoenc1 = trainAutoencoder(TrainImages,Hidden_size1, ...
                'MaxEpochs',400, ...
                'L2WeightRegularization',L2WR, ...
                'SparsityRegularization',SR, ...
                'SparsityProportion',SP, ...
                'ScaleData', false);
            
            %             figure()
            %             plotWeights(autoenc1);
            %             conf = strcat('H=',num2str(Hidden_size1),'  WR=',num2str(L2WR),'  SR=',num2str(SR),'  SP=',num2str(SP));
            %             title(conf)
            feat1 = encode(autoenc1,TrainImages);
            
            Hidden_size2 = 50;
            autoenc2 = trainAutoencoder(feat1,Hidden_size2, ...
                'MaxEpochs',100, ...
                'L2WeightRegularization',L2WR, ...
                'SparsityRegularization',SR, ...
                'SparsityProportion',SP, ...
                'ScaleData', false);
            
            %             plotWeights(autoenc2);
            %             conf = strcat('H=',num2str(Hidden_size1),'  WR=',num2str(L2WR),'  SR=',num2str(SR),'  SP=',num2str(SP));
            %             title(conf)
            feat2 = encode(autoenc2,feat1);
            
            intra_class=zeros(1,600);
            k=1;
            for pointer1=0:6:234
                for i= 1:1:6
                    for j=2:1:6
                        if i<j
                            intra_class(1,k)=(mse(feat2(:,i+pointer1),feat2(:,j+pointer1)));
                            k=k+1;
                        end
                    end
                end
            end
            k1=1;
            inter_class=zeros(1,28080);
            for pointer1=0:6:234
                for pointer2=0:6:234
                    if(pointer1<pointer2)
                        for i=1:1:6
                            for  j=1:1:6
                                inter_class(1,k1)=(mse(feat2(:,i+pointer1),feat2(:,j+pointer2)));
                                k1=k1+1;
                            end
                        end
                    end
                end
            end
            
            ratio=sum(inter_class(:))/sum(intra_class(:));
            H=[(-1*inter_class) (-1*intra_class)];
            T=[zeros(1,28080),ones(1,600)];
            H_mat=ezroc3(H,T,2,' ',1);
        end
    end
end
