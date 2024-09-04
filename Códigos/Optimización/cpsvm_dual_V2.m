function [Prediction,Tf,S] = cpsvm_dual_V2(data, labels, Xtest, FunPara)

[num, ~] = size(data);
C1=FunPara.C1;
C2=FunPara.C2;
epsi=FunPara.epsi;
kerfPara = FunPara.kerfPara;

% Compute Kernel
if strcmp(kerfPara.type,'lin')
    K = data*data';
else
    K=kernelfun(data,kerfPara);
end

t0=cputime;
cvx_begin
cvx_quiet true
cvx_precision('low')
    variable alpha_val(num);
    variable beta_val(num);
    variable gamma_val(num);
    dual variable b
    maximize(sum(0.5 * alpha_val .* (labels+epsi) - gamma_val) - 0.5 * sum(sum((C2*labels+labels .* alpha_val + beta_val - gamma_val)' * K * (C2*labels+labels .* alpha_val + beta_val - gamma_val))));
    subject to
    b:   sum(C2*labels+labels .* alpha_val + beta_val - gamma_val) ==0;
 %    sum(C2*labels+labels .* alpha_val + beta_val - gamma_val) ==0;
        0 <= alpha_val <= C1 / epsi;
        beta_val >= 0;
        gamma_val >= 0;
cvx_end
Tf=cputime-t0;
b=-b;

% % Obtener los índices de los vectores de soporte
%support_indices = (alpha_val > 6e-3) & (alpha_val < (C1/epsi-1e-3));
%b = 0.5 +mean(0.5*epsi*labels(support_indices) - K(support_indices, :) * (C2*labels+alpha_val .* labels + beta_val - gamma_val));

if strcmp(kerfPara.type,'lin')
   w=data'*(C2*labels+alpha_val .* labels + beta_val - gamma_val);
   Val_Xt=Xtest*w + b;
   S.w=w;                                                                             
else
   Kt=kernelfun(data,kerfPara,Xtest);
   Val_Xt=Kt'*(C2*labels+alpha_val .* labels + beta_val - gamma_val)+b;
end
Prediction=sign(Val_Xt-0.5);

S.b=b;
S.alpha=alpha_val;
S.beta=beta_val;
S.gamma=gamma_val;
S.Prob=Val_Xt;


