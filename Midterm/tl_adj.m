function [M,M_tr]=tl_adj(dt,sigma,beta,rho,future_f)

%--------------------------------------
% adjoint of the lorenz 63 model with a modified Euler scheme
%
% M:     tangent-linear model of lorenz-63
% M_tr:  adjoint model of lorenz-63
% 
% sigma: lorenz-63 model parameter (= A)
% beta:  lorenz-63 model parameter (= B)
% rho:   lorenz-63 model parameter (= C)
%
% Original by A. Fowler
% Modified by T. Matsuo for the course
%--------------------------------------

x=future_f(1);
y=future_f(2);
z=future_f(3);

%firstly calc df1
dx1=dt*sigma*(y-x);
dy1=dt*(x*(rho-z)-y);
dz1=dt*(x*y-beta*z);

%-----------

% define A=(M_1') 
GA(1,1)=-dt*sigma;
GA(2,1)=dt*(rho-z);
GA(3,1)=dt*(y);
GA(1,2)=dt*sigma;
GA(2,2)=-dt;
GA(3,2)=dt*(x);
GA(1,3)=0;
GA(2,3)=-dt*(x);
GA(3,3)=-dt*beta;

% define B=M_half' (transform from f_hat_half to 2*f_hat_1)
GB(1,1)=-dt*sigma;
GB(2,1)=dt*(rho-(z+dz1));
GB(3,1)=dt*(y+dy1);
GB(1,2)=dt*sigma;
GB(2,2)=-dt;
GB(3,2)=dt*(x+dx1);
GB(1,3)=0;
GB(2,3)=-dt*(x+dx1);
GB(3,3)=-dt*beta;

%
Id=eye(3);

M=Id+0.5*(GB*(GA+Id)+GA);
M_tr=M';
