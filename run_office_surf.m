 clc;
clear all;
close all;
addpath libsvm-new

src_str = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgt_str = {'amazon','webcam','dslr','Caltech10','webcam','dslr','amazon','Caltech10','dslr','amazon','Caltech10','webcam'};
load('para/office_surf_para.mat');

%%
total_num = 0;
for i = 1:length(tgt_str)
    alpha(i)=0.001;
    src = src_str{i};
    tgt = tgt_str{i};

    fprintf(' %s vs %s ', src, tgt);
    load(['data/office+caltech_10 SURF/' src '_SURF_L10.mat']); 
    Xs = fts';
    Xs_label = labels;
    clear fts;
    clear labels;

    load(['data/office+caltech_10 SURF/' tgt '_SURF_L10.mat']); 
    Xt = fts';
    Xt_label = labels;
    clear fts;
    clear labels;
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
end

mean_num = total_num / 12;
fprintf('mean = %0.2f\n', mean_num);
