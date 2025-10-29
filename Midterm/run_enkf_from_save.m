function run_enkf_from_save(Ne, infl)
% RUN_ENKF_FROM_SAVE
% Ensemble Kalman Filter (stochastic, perturbed observations) for Lorenz-63.
%
% Usage: run_enkf_from_save(50, 1.02)
%
% Inputs:
%   Ne   - ensemble size (default = 50)
%   infl - multiplicative inflation factor (default = 1.02)
%
% Assumes the saved .mat contains:
%   A,B,C, dt, ns, T, fT, xtrue, y, Q, R, Bcov, xb

clear all; clc;

Ne = 50; % NUMBER OF ENSEMBLES
infl = 1.0; % INFLATION

%% -------- Load .mat file from saves ----------
saveDir = 'saves';
[file, path] = uigetfile(fullfile(saveDir, '*.mat'), 'Select a save file to load');
if isequal(file, 0)
    error('No .mat file selected or found in %s.', saveDir);
end
matPath = fullfile(path, file);
S = load(matPath);
[~, baseName, ~] = fileparts(matPath);
fprintf('Loaded workspace: %s\n', matPath);

% Unpack parameters
A = S.A; B = S.B; C = S.C;
dt = S.dt; ns = S.ns;
T  = S.T;  fT = S.fT;
xtrue = S.xtrue;
y      = S.y;
Q      = S.Q; R = S.R;
Bcov   = S.Bcov;
xb     = S.xb;

nx = 3; H = eye(3);
nSteps = T + fT;

% bookkeeping
isObsStep = false(1, nSteps);
obsIndexAtStep = zeros(1, nSteps);
k = 0;
for t = ns:ns:min(T, nSteps)
    k = k + 1;
    isObsStep(t) = true;
    obsIndexAtStep(t) = k;
end

%% -------- Ensemble initialization ----------
rng('default');
En = mvnrnd(xb, Bcov, Ne)';  % 3xNe ensemble
xa_mean = zeros(nx, nSteps);
xa_mean(:,1) = mean(En,2);

%% -------- EnKF Loop ----------
for t = 1:(nSteps-1)
    % ---- Forecast ----
    for i = 1:Ne
        En(:,i) = modeuler(dt, En(:,i), A, B, C) + mvnrnd(zeros(nx,1), Q);
    end

    % Inflation (optional)
    En = mean(En,2) + infl*(En - mean(En,2));

    % ---- Analysis (if obs available) ----
    if isObsStep(t+1)
        j = obsIndexAtStep(t+1);
        y_obs = y(:,j);

        % Ensemble mean and anomalies
        x_mean = mean(En,2);
        Aens = En - x_mean;

        % Sample covariances
        Pf = (Aens * Aens') / (Ne-1);
        Syy = H * Pf * H' + R;
        K = Pf * H' / Syy;

        % Perturbed observations (stochastic EnKF)
        Y_pert = y_obs + mvnrnd(zeros(size(y_obs)), R, Ne)';

        % Update each ensemble member
        for i = 1:Ne
            innov = Y_pert(:,i) - H * En(:,i);
            En(:,i) = En(:,i) + K * innov;
        end
    end

    xa_mean(:,t+1) = mean(En,2);
end

%% -------- Diagnostics ----------
error = xa_mean - xtrue;
rmse_t = sqrt(mean(error.^2, 1));

% NEES / NIS (approx from ensemble spread) - THIS IS WRONG
Pf_diag = var(En,0,2);
NEES = sum((error.^2) ./ Pf_diag,1);
NIS = NEES; % same structure since H=I here

dof = nx;
alpha = 0.95;
nees_lower = chi2inv(0.025, dof);
nees_upper = chi2inv(0.975, dof);
consistency = mean(NEES >= nees_lower & NEES <= nees_upper, 'omitnan') * 100;

%% -------- Plots ----------

%% -------- State Trajectories with ±2σ bands ----------

tgrid = (1:nSteps) * dt;

% Observation times
t_obs = (ns:ns:T) * dt;

confK = 2;
sigX = std(En(1,:),0,2);
sigY = std(En(2,:),0,2);
sigZ = std(En(3,:),0,2);

figure('Name','State Trajectories (EnKF)');
for i=1:3
    subplot(3,1,i); hold on;
    plot(tgrid, xtrue(i,:), 'k-', 'LineWidth',2.5);
    plot(tgrid, xa_mean(i,:), '-', 'LineWidth',2);
    fill([tgrid fliplr(tgrid)], [xa_mean(i,:)+confK*sigX fliplr(xa_mean(i,:)-confK*sigX)], ...
        [0.8 0.9 1], 'EdgeColor','none', 'FaceAlpha',0.5);
    plot(t_obs, y(i,:), 'r*', 'MarkerSize', 4);                        % observations
    ylabel(['x_',num2str(i)]);
    if i==1, title('State vs Time (Truth, EnKF mean, ±2σ)'); end
    if i==3, xlabel('Time'); end
    grid on;
end

%% ------- Metrics ---------
figure('Name','EnKF Metrics');
subplot(3,1,1);
plot(tgrid, rmse_t, 'LineWidth',1.4);
xlabel('Time'); ylabel('RMSE');
title(sprintf('RMSE vs Time — mean %.3f', mean(rmse_t,'omitnan')));
grid on;

subplot(3,1,2);
plot(tgrid, NEES, 'b-', 'LineWidth', 1.2); hold on;
yline(nees_lower, '--g', 'Lower 95%');
yline(nees_upper, '--r', 'Upper 95%');
title(sprintf('NEES vs Time — %.2f%% Consistent', consistency));
ylim([0,15]); grid on;

subplot(3,1,3);
plot(tgrid, NIS, 'm-', 'LineWidth',1.2); hold on;
yline(nees_lower, '--g', 'Lower 95%');
yline(nees_upper, '--r', 'Upper 95%');
xlabel('Time'); ylabel('NIS');
ylim([0,15]); grid on;

%% -------- Save results ----------
outFile = fullfile(saveDir, sprintf('%s_enkf_Ne%d.mat', baseName, Ne));
save(outFile, 'xa_mean', 'En', 'rmse_t', 'NEES', 'NIS', 'Ne', 'infl');
fprintf('Saved EnKF results to %s\n', outFile);

end
