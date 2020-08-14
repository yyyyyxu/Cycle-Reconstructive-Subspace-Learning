function  [P,Zs,Zt] = CRSL(iter_num,Xs,Xt,Xs_label,Xt_label,alpha,beta,lambda,delta)



Y = Construct_Y(Xs_label,length(Xs_label)); 
B = Construct_B(Y);
Class = length(unique(Xs_label));
Max_iter = 100;
[m,n1] = size(Xs); n2 = size(Xt,2); %n2 = nt n1=ns
max_mu = 10^6;
mu = 0.1;
rho = 1.01;
convergence = 10^-6;
options = [];
options.ReducedDim = Class;
[P1,~] = PCA1(Xs', options);
% ----------------------------------------------
%               Initialization
% ----------------------------------------------
M = ones(Class,n1);
P = zeros(m,Class);
P_n = zeros(m,Class);

Zs = zeros(n1,n2);
Zt = zeros(n1,n2);

Z1 = zeros(n1,n2);
Z2 = zeros(n1,n2); % Z2 = inv(alpha*D+mu*eye(n2))*(Y4+mu*Zs)

D = zeros(n2,n2);
F = zeros(n2,n1); 
F_n = zeros(n2,n1); 

Y1 = zeros(Class,n2);
Y2 = zeros(n1,Class);
Y3 = zeros(n1,n2);
Y4 = zeros(n1,n2);
% ------------------------------------------------
%                   Main Loop
% ------------------------------------------------
for iter = 1:iter_num
     
   
    % updating P
    if (iter == 1)
        P = P1;
    else
        C = Y+B.*M;
        V1 = 2*delta*X*BB*X'+(1+mu)*Xs*Xs'+mu*Xs*Zs*Zs'*Xs'-mu*Xt*Zs'*Xs'-mu*Xs*Zs*Xt'+mu*Xt*Xt'-mu*Xs*Zt*Xt'-mu*Xt*Zt'*Xs'+mu*Xt*Zt'*Zt*Xt'+lambda*eye(m);
        V2 = Xs*C'-Xs*Zs*Y1'+Xt*Y1'-Xs*Y2+Xt*Zt'*Y2;
        P = V1\V2;
    end       
    
    % updating M
    R = P'*Xs-Y;
        gp = R.*B;
    [numm1,numm2] = size(gp);
    for jk1 = 1:numm1
        for jk2 = 1:numm2
            M(jk1,jk2) = max(gp(jk1,jk2),0);
        end
    end       
    
    % updating Zt
    V3 = Y2*P'*Xt-Y3+mu*Xs'*P*P'*Xt+mu*Z1;
    V4 = mu*Xt'*P*P'*Xt+mu*eye(n2);
    Zt = V3/V4;   
    Zt_cov(iter)=norm(Zt,2);
    
    % updating Zs
    V5 = mu*Xs'*P*P'*Xs+mu*eye(n1);
    V6 = -Xs'*P*Y1-Y4+mu*Xs'*P*P'*Xt+mu*Z2;
    Zs =  V5\V6;   
    Zs_cov(iter)=norm(Zs,2);
    
    % updating  Z1
    ta = beta/mu;
    temp_Z1 = Zt+Y3/mu;
    [U01,S01,V01] = svd(temp_Z1,'econ');
    S01 = diag(S01);
    svp = length(find(S01>ta));
    if svp >= 1
        S01 = S01(1:svp)-ta;
    else
        svp = 1;
        S01 = 0;    
    end
    Z1 = U01(:,1:svp)*diag(S01)*V01(:,1:svp)';
    
    % updating Z2 
    taa = alpha/mu;
    temp_Z2 = Zs+Y4/mu;
    Z2 = max(0,temp_Z2-taa)+min(0,temp_Z2+taa);
    
    X_train_f = P'*Xs;
    Y_test_f  = P'*Xt; 
    
    tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
    model_f1 = svmtrain(Xs_label, X_train_f', tmd);
    [Xt_label_f, acc] = svmpredict(Xt_label, Y_test_f', model_f1);
    acc = acc(1);  
    

    X = [Xs,Xt];
    Yall = [Xs_label;Xt_label_f];
    %--------------------------------------
    % construct distance between each class
    BBT1=[];

    for c = reshape(unique(Xs_label),1,Class)
        SClassNum(c)=length(find(Xs_label==c));
        TClassNum(c)=length(find(Xt_label_f==c));
        ClassNum(c)=length(find(Yall==c));
    end

    BBS1=zeros(n1,n1);
    for i=1:n1
        BBS1(i,i) = n1/SClassNum(Xs_label(i))*(SClassNum(Xs_label(i)));
        for j=i+1:n1
            if Xs_label(i)==Xs_label(j)
                BBS1(i,j) = n1/SClassNum(Xs_label(i))*(-1);
                BBS1(j,i) = n1/SClassNum(Xs_label(i))*(-1);
            end
        end
    end

    YsColumn=singlelbs2multilabs(Xs_label,Class);
    BBS2 = YsColumn*YsColumn' - ones(n1,n1);

    for i=1:n1
        BBS2(i,i) = n1 - SClassNum(Xs_label(i));
    end
    BBS1=BBS1-rho*BBS2;


    if ~isempty(Xt_label_f) && length(Xt_label_f)==n2
        BBT1=zeros(n2,n2);
        for i=1:n2
                BBT1(i,i) = n2/TClassNum(Xt_label_f(i))*(TClassNum(Xt_label_f(i)));
            for j=i+1:n2
                if Xt_label_f(i)==Xt_label_f(j)
                	BBT1(i,j) = n2/TClassNum(Xt_label_f(i))*(-1);
                	BBT1(j,i) = n2/TClassNum(Xt_label_f(i))*(-1);
                end
            end
        end
        BBT1 = BBT1;

        Yt0Column=singlelbs2multilabs(Xt_label_f,Class);
        BBT2 = Yt0Column*Yt0Column' - ones(n2,n2);

        for i=1:n2
            BBT2(i,i) = n2 - TClassNum(Xt_label_f(i));
        end
        BBT1=BBT1-rho*BBT2;
    end

    BB = blkdiag(BBS1,BBT1);
    BB = BB/norm(BB,'fro');
    %-------------------------------------------  
    %%
    % updating Y1, Y2, Y3,Y4
	Y1 = Y1+mu*(P'*Xs*Zs-P'*Xt);
	Y2 = Y2+mu*(Xs'*P-Zt*Xt'*P);
    Y3 = Y3+mu*(Zt-Z1);
    Y4 = Y4+mu*(Zs-Z2);
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
    % checking convergence
    leq1 = norm(P'*Xs*Zs-P'*Xt,Inf);
    leq2 = norm(Xs'*P-Zt*Xt'*P,Inf);
    leq3 = norm(Zt-Z1,Inf);
    leq4 = norm(Zs-Z2,Inf);
    leq_P = norm(P - P_n,1)/norm(P,1);
    if iter > 2
         if (leq1<convergence && leq2<convergence && leq3<convergence && leq4<convergence )|| (leq_P < convergence)
              break
         end
    end
    P_n = P;
end
iter;
end

function B = Construct_B(Y)
%%
B = Y;
B(Y==0) = -1;
end

function Y = Construct_Y(gnd,num_l)
%%
% gnd:Label vector£»
% num_l:Represents the number of samples with labels£»
% Y:The resulting label matrix£»
nClass = length(unique(gnd));
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        end  
    end
end
end


function label=singlelbs2multilabs(y,nclass)
    L=length(y);
    label=zeros(L,nclass);
    for i=1:L
        label(i,y(i))=1;
    end
end