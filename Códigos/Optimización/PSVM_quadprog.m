% Solving the following Quadratic Problems with quadprog function

% Dual problem of PSVM model
%  minimize  0.5*x'*Q*x + f'*x
%  subject to  A*x=0
%          0<= x <= Cv
% with x=[alpha,beta,gamma] in R^{3m}
%      b=[-0.5*Y'-0.5*epsi*e', 0, e'] in R^{3m}
%      A=[Y', e', -e'] in R^{3m}

function [Prediction,Tf,S]=PSVM_quadprog(X,Y,Xtest,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%

%       Input:
%               X       - Training Data matrix (Each row vector is a data point)
%               Y       - Training label vector
%               Xtest      - Test Data matrix.
%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   c: [0,inf] Parameter to tune the weight.
%                   epsi: (0,1] Parameter to tune

%       Output:
%               Ytest  - Predict value of the Xt.
%              Val_Xt - Value of the Xt

C=FunPara.C;
epsi=FunPara.epsi;
kerfPara = FunPara.kerfPara;
[m, ~] = size(X);
e=ones(m,1);

% Compute Kernel
if strcmp(kerfPara.type,'lin')
    K = X*X';
else
    K=kernelfun(X,kerfPara);
end
K1=K.*(Y*Y');
K2=K.*Y; %D*K
K3=K*diag(Y); % K*D

Q=[K1,K2,-K2;K3, K, -K; -K3, -K, K];
Q=(Q+Q')/2;
Q=Q+1.e-8*eye(3*m);
f=[-0.5*Y-0.5*epsi*e; zeros(m,1); e];
Ae=[Y', e', -e'];
Cu=[(C/epsi)*e;Inf*e;Inf*e];

t0=cputime;
[sol,fval,~,~,lambda]= quadprog(Q,f,[],[],Ae,0,zeros(3*m,1),Cu); 
Tf=cputime-t0;

S.alpha=sol(1:m);
S.beta=sol(m+1:2*m);
S.gamma=sol(2*m+1:3*m);
S.b=lambda.eqlin;

if strcmp(kerfPara.type,'lin')
   w=X'*(S.alpha .* Y + S.beta - S.gamma);
   Prob_Xt=Xtest*w + S.b;
   S.w=w;
else
   Kt=kernelfun(X,kerfPara,Xtest);
    Prob_Xt=Kt'*(S.alpha .* Y + S.beta - S.gamma)+S.b;
end
Prediction=sign(Prob_Xt-0.5);
S.Prob=Prob_Xt;



