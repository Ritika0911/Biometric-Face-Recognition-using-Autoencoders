clc;
clear all;
close all;
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
        %img = imresize(img,0.5);
        img = double(img)/256;
        Train_cell = mat2cell(img,112,92);
        TrainImages = [Train_cell TrainImages];
        
    end
    
    for k = 7:Images_per_class
        filename = [num2str(j) '.pgm'];
        img =imread(filename);
        %img = imresize(img,0.5);
        img = double(img)/256;
        Test_cell = mat2cell(img,112,92);
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

autoenc1 = trainAutoencoder(TrainImages,Hidden_size1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.006, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.30, ...
    'ScaleData', false);

feat1 = encode(autoenc1,TrainImages);

Hidden_size2 = 50;
autoenc2 = trainAutoencoder(feat1,Hidden_size2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.006, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

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

%% Train Softmax Layer
softnet = trainSoftmaxLayer(feat2,T_Train_Images,'MaxEpochs',400);

%% Forming a stacked neural network
% view(autoenc1)
% view(autoenc2)
% view(softnet)

deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)

%%
% Get the number of pixels in each image
imageWidth = 112;
imageHeight = 92;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(TestImages));
for i = 1:numel(TestImages)
    xTest(:,i) = TestImages{i}(:);
end

%%
% You can visualize the results with a confusion matrix. The numbers in the
% bottom right-hand square of the matrix give the overall accuracy.
y = deepnet(xTest);
ezroc3(y,T_Test_Images,2,'Before tuning',1)
error_test = mse(deepnet, T_Test_Images, y);

%% Fine tuning the deep neural network
% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(TrainImages));
for i = 1:numel(TrainImages)
    xTrain(:,i) = TrainImages{i}(:);
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,T_Train_Images);

%%
% You then view the results again using a confusion matrix.
y = deepnet(xTest);
ezroc3(y,T_Test_Images,2,'After tuning',1);
