function [Prediction,Tf,S] = psvm_dual(data, labels, Xtest, FunPara)

C=FunPara.C;
eps=FunPara.epsi;
kerfPara = FunPara.kerfPara;
[num, ~] = size(data);

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
    maximize(sum(0.5 * alpha_val .* (labels + eps) - gamma_val) - 0.5 * sum(sum((labels .* alpha_val + beta_val - gamma_val)' * K * (labels .* alpha_val + beta_val - gamma_val))));
    
    subject to
   b:     sum(labels .* alpha_val + beta_val - gamma_val) == 0;
        0 <= alpha_val <= C / eps;
        beta_val >= 0;
        gamma_val >= 0;
cvx_end
Tf=cputime-t0;
b=-b;

if strcmp(kerfPara.type,'lin')
   w=data'*(alpha_val .* labels + beta_val - gamma_val);
   Val_Xt=Xtest*w + b;
   S.w=w;                                                                             
else
   Kt=kernelfun(data,kerfPara,Xtest);
   Val_Xt=Kt'*(alpha_val .* labels + beta_val - gamma_val)+b;
end
Prediction=sign(Val_Xt-0.5);

S.b=b;
S.alpha=alpha_val;
S.beta=beta_val;
S.gamma=gamma_val;
S.Val_Xtest=Val_Xt;

% Obtener los índices de los vectores de soporte
%support_indices = (alpha_val > 1.e-5) & (alpha_val < (C/eps-5.e-5));
% Se calcula el bias, tomando el promedio
%b =0.5+ mean(0.5*eps*labels(support_indices)  - K(support_indices, :) * (alpha_val .* labels + beta_val - gamma_val))
end
