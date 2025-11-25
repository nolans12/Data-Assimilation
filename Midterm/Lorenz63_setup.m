close all;
%-------------------------------------------------
% PROGRAM: Filtering Experiemnt Setup for the Lorenz-63  
%
%-------------------------------------------------
% MODEL SETUP (Do not change)
N  = 3; %total number of variables
MO = 3; % Number of observations at analysis time
randn('state',0) % random seed
%Lorenz-63 model parameters
A = 10.;
B = 28.;
C = 8./3.; 
%------------------------ 
% EXPERIMENT SET-UP
xstr='True value of x at t=0';
ystr='True value of y at t=0';
zstr='True value of z at t=0';
astr={'1.5','-1.5','25.5'}; % default input parameters
true_str=inputdlg({xstr,ystr,zstr},'Initial data',1,astr);
truex=str2num(true_str{1});
truey=str2num(true_str{2});
truez=str2num(true_str{3});
%-----------------------
xstr='Length of assimilation window';
x2str='Length of forecast';
ystr='Time step';
zstr='Frequency of observations (in time steps)';
astr={'3','1','0.01','20'};  % default input parameters
temp_str=inputdlg({xstr,x2str,ystr,zstr},'Run information',1,astr);
T=str2num(temp_str{1});
fT=str2num(temp_str{2});
dt=str2num(temp_str{3});
ns=str2num(temp_str{4});
T=T/dt;
fT=fT/dt;
%-----------------------
xstr='Observation error variance';
astr={'1'}; % suggested input parameters
temp_str=inputdlg({xstr},'',1,astr);
varobs=str2num(temp_str{1});
%-----------------------
xstr='Background error variance';
astr={'1.'}; % suggested input parameters
temp_str=inputdlg({xstr},'',1,astr);
varb=str2num(temp_str{1});
sb=sqrt(varb);
B_structure=[1,0,0;0,1,0;0,0,1];
Bcov=varb*B_structure;
%------------------------
% SAVE STRING FOR MAT FILE
savestr='String for .mat file 1';
astr={'default'}; % default input parameters
save_str=inputdlg({savestr},'String for .mat file 2',1,astr);
%-----------------------
%Set up the model error covariance (e.g., process noise), from Evensen, 1997.
  Q(1,1)=0.1491;
  Q(1,2)=0.1505;
  Q(1,3)=0.0007;
  Q(2,1)=0.1505;
  Q(2,2)=0.9048;
  Q(2,3)=0.0014;
  Q(3,1)=0.0007;
  Q(3,2)=0.0014;
  Q(3,3)=0.9180;
%-----------------------
%COMPUTE THE TRUTH
x(1,1) =truex;
x(2,1) =truey ;
x(3,1) =truez ;
%The true value at t=0 plus a random error
svar=sqrt(1.);
xs(1,1) = truex + svar * randn(1,1);
xs(2,1) = truey + svar * randn(1,1);
xs(3,1) = truez + svar * randn(1,1);
for t=2:T+fT
  x(:,t)=modeuler(dt,x(:,t-1), A, B, C);
end
xtrue = x;
%-----------------------_
%GENERATE SYNTHETIC OBSERVATIONS
NM = floor(T/ns);  %number of measurement times
y = zeros(MO,NM);
sobs=sqrt(varobs);
R_structure=[1,0,0.;0.,1,0.;0.,0.,1];
  R=varobs*R_structure;
  [V,L] = eig(R);
  Msqrt_cov=V*sqrt(L);
  M_cov = Msqrt_cov*Msqrt_cov';
for t = 1:NM     %number of  measurement times
   %the truth plus ob error
   y(:,t) = xtrue(:,t*ns) + Msqrt_cov* randn(3,1); 
end
%-----------------------
% CALCULATE PRIOR GUESS AT INITIAL TIME
[V,L] = eig(Bcov);
Bsqrt_cov=V*sqrt(L);
xb=xtrue(:,1) + Bsqrt_cov*randn(3,1); 
%---------------------------------
%PLOT
%plot the truth
subplot(3,1,1)
title('Truth')
hold on
plot([1:T+fT]*dt,xtrue(1,:),'k')
axis([0 (T+fT)*dt -20 20]);
ylabel('X')
subplot(3,1,2)
hold on
plot([1:T+fT]*dt,xtrue(2,:),'k')
axis([0 (T+fT)*dt -25 25]);
ylabel('Y')
subplot(3,1,3)
hold on
plot([1:T+fT]*dt,xtrue(3,:),'k')
axis([0 (T+fT)*dt 5 45]);
xlabel('time')
ylabel('Z')
% plot observations
subplot(3,1,1)
plot((ns:ns:T)*dt,y(1,:),'r*')
subplot(3,1,2)
plot((ns:ns:T)*dt,y(2,:),'r*')
subplot(3,1,3)
plot((ns:ns:T)*dt,y(3,:),'r*')


% SAVE WORKSPACE TO .mat FILE
saveDir = 'saves'; 
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end
fileName = [save_str{1} '.mat']; 
fullPath = fullfile(saveDir, fileName);
save(fullPath); % save entire workspace
fprintf('Workspace successfully saved to: %s\n', fullPath);



