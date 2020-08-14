clc;
clear all;
close all;
addpath libsvm-new
src_str = {'COIL_1','COIL_2'};
tgt_str = {'COIL_2','COIL_1'};

load('para/COIL_para.mat');

total_num = 0;
for i = 1:length(src_str)
     
    alpha(i)=0.001;
    
    src = src_str{i};
    tgt = tgt_str{i};

    fprintf(' %s vs %s ', src, tgt);

    load(['data/COIL/' src '.mat']); 
    Xs = X_src;
    Xs_label = Y_src;
    Xt = X_tar;
    Xt_label = Y_tar;
    clear X_src;
    clear Y_src;
    clear X_tar;
    clear Y_tar;
    % ------------------------------------------
    %             Transfer Learning
    % ------------------------------------------
    Xs = zscore(Xs',1);
    Xs = normr(Xs)';
    Xs = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]);  %//normalization
    Xt = zscore(Xt',1);
    Xt = normr(Xt)';
    Xt = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);  %//normalization 
    
    
    iter_num = 10;
    [P,Zs,Zt] = CRSL(iter_num,Xs,Xt,Xs_label,Xt_label,alpha(i),beta(i),lambda(i),delta(i));
    X_train = P'*Xs;
    Y_test  = P'*Xt; 
    % -------------------------------------------
    %               Classification
    % -------------------------------------------
    %%
	tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
    model = svmtrain(Xs_label, X_train', tmd);
    [~, acc] = svmpredict(Xt_label, Y_test', model);
    acc = acc(1);  
    total_num = total_num + acc;
	fprintf('acc = %0.2f\n',acc);
%     end
end
mean_num = total_num / 2;
fprintf('mean = %0.2f\n', mean_num);
