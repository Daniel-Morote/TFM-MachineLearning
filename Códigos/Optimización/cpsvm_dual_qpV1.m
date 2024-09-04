% Solving the following Quadratic Problems with quadprog function

% Dual problem of CPSVM model (version 1)
%  minimize  0.5*x'*Q*x + f'*x
%  subject to  A*x=b
%          0<= x <= Cv
% with x=[alpha,beta,gamma] in R^{3m}
%      f=[-0.5*epsi*Y; 0; e] + C2*[D*K*Y;K*Y;-K*Y] in R^{3m}
%      A=[Y', e', -e'] in R^{3m}
%      b=-C2*Y'*e

function [Prediction,Tf,S]=cpsvm_dual_qpV1(X,Y,Xtest,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%

%       Input:
%               X       - Training Data matrix (Each row vector is a data point)
%               Y       - Training label vector
%               Xtest      - Test Data matrix.
%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   C1,C2: [0,inf] Parameter to tune the weight.
%                   epsi: (0,1] Parameter to tune

%       Output:
%               Ytest  - Predict value of the Xt.
%              Val_Xt - Value of the Xt

C1=FunPara.C1;
C2=FunPara.C2;
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

f1=[-0.5*epsi*Y; zeros(m,1); e];
Ky=K*Y;
f2=[diag(Y)*Ky;Ky;-Ky];
f=f1+C2*f2;
Ae=[Y', e', -e'];
be=-C2*sum(Y);
Cu=[(C1/epsi)*e;Inf*e;Inf*e];

t0=cputime;
[sol,fval,exitflag,output,lambda]= quadprog(Q,f,[],[],Ae,be,zeros(3*m,1),Cu); 
Tf=cputime-t0;

S.alpha=sol(1:m);
S.beta=sol(m+1:2*m);
S.gamma=sol(2*m+1:3*m);
S.b=lambda.eqlin;

if strcmp(kerfPara.type,'lin')
   w=X'*(C2*Y+S.alpha .* Y + S.beta - S.gamma);
   Prob_Xt=Xtest*w + S.b;
   S.w=w;
else
   Kt=kernelfun(X,kerfPara,Xtest);
   Prob_Xt=Kt'*(C2*Y+S.alpha .* Y + S.beta - S.gamma)+S.b;
end
Prediction=sign(Prob_Xt-0.5*epsi);
S.Prob=Prob_Xt;


