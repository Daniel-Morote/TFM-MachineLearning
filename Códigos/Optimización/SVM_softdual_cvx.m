function [Ytest,Tf,Sol]=SVM_softdual_cvx(X,Y,Xt,FunPara)
%       Input:
%               X           - Training data matrix (Each row vector is a data point).
%               Y           - Training Label data (-1 or 1)
%               Xt       - Test Data matrix (Each row vector is a data point).
%
%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   c: [0,inf] Parameter to tune the weight.

%                   kerfPara:Kernel parameters. See kernelfun.m.
%
%       Output:
%               Ytest - Predict value of the TestX.
%
%       Example:
%           A = rand(50,10);
%           B = rand(60,10);
%           X=[A;B];
%           Y=[ones(50,1);-ones(60,1)];
%           TestX=rand(20,10);
%           FunPara.c=2^(2);
%           Ytest=SVM_softdual_cvx(X,Y,TestX,FunPara);

C=FunPara.c;
kerfPara = FunPara.kerfPara;
% Compute Kernel
if strcmp(kerfPara.type,'lin')
   K = X*X';
else
   K=kernelfun(X,kerfPara);
end
K=K.*(Y*Y');

[n, ~] = size(X);
t0=cputime;
cvx_begin quiet
cvx_precision('low')
cvx_solver sedumi
variable z(n);
dual variable b
    maximize(sum(z) - z' * K * z/2);
    subject to
    0<= z <= C;
    b: Y' * z == 0;
cvx_end
Tf=cputime-t0;

bias=-b;
Sol.alpha=z;
Sol.b=bias;
%  sv_ind = z > 1e-5 & z < C - 1e-4;
%  b = mean(labels(sv_ind) - data(sv_ind,:) * w);
if strcmp(kerfPara.type,'lin')
    w=X'*(z.*Y);
    Sol.Val_Xt=Xt*w+bias;
    Sol.w=w;
else
    Kt=kernelfun(X,kerfPara,Xt);
    Sol.Val_Xt=Kt'*(z.*Y)+bias;
end
Ytest=sign(Sol.Val_Xt);

sv_indx=z > 1e-5;
end
