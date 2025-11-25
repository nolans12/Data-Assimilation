function run_enkf_from_save()
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
Q(1,1) = Q(2,2); % to fix tuning via rank histograms
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
En_history = zeros(3, Ne, nSteps);
En_history(:,:,1) = En;
xa_mean = zeros(nx, nSteps);
Pa = zeros(nx, nx, nSteps);
xa_mean(:,1) = mean(En,2);
Pa(:,:,1) = Bcov;

NEES = nan(nx, nSteps);         % NEES metric
error    = nan(nx, nSteps);   % xa - truth (for metrics)

%% -------- EnKF Loop ----------
for t = 1:(nSteps-1)
    % ---- Forecast all ensembles ----
    for i = 1:Ne
        En(:,i) = modeuler(dt, En(:,i), A, B, C) + mvnrnd(zeros(nx,1), Q);
    end

    % save history for forecasted for rank histograms
    En_history(:,:,t+1) = En;

    % Inflation mmaybe
    En = mean(En,2) + infl*(En - mean(En,2));

    % ---- UPdate (if obs step) ----
    if isObsStep(t+1)
        j = obsIndexAtStep(t+1);
        y_obs = y(:,j);

        % Ensemble mean and anomalies
        x_mean = mean(En,2);
        Aens = En - x_mean;

        % Sample covariance across all ensembles given the error from mean
        Pf = (Aens * Aens') / (Ne-1);

        % Now gain update - using the sample  cov estimate 
        Syy = H * Pf * H' + R;
        K = Pf * H' / Syy;

        % Perturbed observations - or D?
        Y_pert = y_obs + mvnrnd(zeros(size(y_obs)), R, Ne)';

        % Update each ensemble member
        for i = 1:Ne
            innov = Y_pert(:,i) - H * En(:,i);
            En(:,i) = En(:,i) + K * innov;
        end
    end

    % New estimate is mean of the ensemble
    xa_mean(:,t+1) = mean(En,2);

    % just for post (not used in next update) - take the new sample cov of the Ensample 
    Aens_post = En - xa_mean(:,t+1);
    Pa(:,:,t+1) = (Aens_post * Aens_post') / (Ne-1);

    % error vs truth 
    error(:,t+1) = xa_mean(:,t+1) - xtrue(:,t+1);
    % calculate NEES
    NEES(:,t+1) = error(:,t+1)' * inv(Pa(:,:,t+1)) * error(:,t+1);
end

%% -------- Plots ----------

%% -------- State Trajectories with ±2σ bands ----------

tgrid = (1:nSteps) * dt;

% Observation times
t_obs = (ns:ns:T) * dt;

confK = 2;
sigX = sqrt(squeeze(Pa(1,1,:))).';
sigY = sqrt(squeeze(Pa(2,2,:))).';
sigZ = sqrt(squeeze(Pa(3,3,:))).';

figure('Name','State Trajectories (EnKF)');
% ---- x ----
subplot(3,1,1); hold on;
x_upper = xa_mean(1,:) + confK*sigX; 
x_lower = xa_mean(1,:) - confK*sigX;
plot(tgrid, xtrue(1,:), 'k-', 'LineWidth', 2.5);                   % truth
plot(tgrid, xa_mean(1,:), '-', 'LineWidth', 2.5);                       % EKF est
fill([tgrid, fliplr(tgrid)], [x_upper, fliplr(x_lower)], [0.8 0.9 1], ...
    'EdgeColor','none','FaceAlpha',0.65);                          % ±2σ band
plot(t_obs, y(1,:), 'r*', 'MarkerSize', 4);                        % observations
ylabel('x'); title('x, y, z vs time (Truth, EnKF, ±2σ, Observations)');
grid on;

% ---- y ----
subplot(3,1,2); hold on;
y_upper = xa_mean(2,:) + confK*sigY; 
y_lower = xa_mean(2,:) - confK*sigY;
plot(tgrid, xtrue(2,:), 'k-', 'LineWidth', 2.5);
plot(tgrid, xa_mean(2,:), '-', 'LineWidth', 2.5);
fill([tgrid, fliplr(tgrid)], [y_upper, fliplr(y_lower)], [0.8 0.9 1], ...
    'EdgeColor','none','FaceAlpha',0.65);
plot(t_obs, y(2,:), 'r*', 'MarkerSize', 4);
ylabel('y'); grid on;

% ---- z ----
subplot(3,1,3); hold on;
z_upper = xa_mean(3,:) + confK*sigZ; 
z_lower = xa_mean(3,:) - confK*sigZ;
plot(tgrid, xtrue(3,:), 'k-', 'LineWidth', 2.5);
plot(tgrid, xa_mean(3,:), '-', 'LineWidth', 2.5);
fill([tgrid, fliplr(tgrid)], [z_upper, fliplr(z_lower)], [0.8 0.9 1], ...
    'EdgeColor','none','FaceAlpha',0.65);
plot(t_obs, y(3,:), 'r*', 'MarkerSize', 4);
ylabel('z'); xlabel('Time'); grid on
legend({'Truth','EnKF Estimate','EnKF ±2σ','Observations'}, 'Location','best');

%% ------- Metrics ---------

% RMSE across the 3 state components at each time
rmse_t = sqrt(mean(error.^2, 1, 'omitnan'));  % 1 x nSteps

% Degrees of freedom for chi2inv bounds
dof_nees = nx;           % state dimension
alpha = 0.95;
plus_minus = (1 - alpha) / 2;

% Chi-square consistency bounds (lower and upper)
nees_lower = chi2inv(1 - alpha - plus_minus, dof_nees);
nees_upper = chi2inv(alpha + plus_minus,     dof_nees);

nees_series = NEES(1,:);           

% Consistency percentages
nees_in_bounds = mean(nees_series >= nees_lower & nees_series <= nees_upper, 'omitnan') * 100;

% RMSE
figure('Name','EnKF Metrics');
subplot(2,1,1);
plot(tgrid, rmse_t, 'LineWidth',1.4);
xlabel('Time'); ylabel('RMSE');
title(sprintf('RMSE vs Time — mean %.3f', mean(rmse_t,'omitnan')));
grid on;

% NEES
subplot(2,1,2);
plot(tgrid, nees_series, 'b-', 'LineWidth', 1.4); hold on;
yline(nees_lower, '--g', sprintf('Lower 95%% (χ²_{%.2f, dof=%d}) = %.2f', 1-alpha, dof_nees, nees_lower), 'HandleVisibility','off');
yline(nees_upper, '--r', sprintf('Upper 95%% (χ²_{%.2f, dof=%d}) = %.2f', alpha, dof_nees, nees_upper), 'HandleVisibility','off');
xlabel('Time');
ylabel('NEES');
title(sprintf('NEES vs Time — Consistency bounds [%.2f, %.2f].\n %.2f Consistent.', nees_lower, nees_upper, nees_in_bounds));
ylim([0,15])
grid on;

%% -------- Rank Histograms (compare to observations at obs times) --------
fprintf('Computing rank histograms vs observations using En_history...\n');

nBins = Ne + 1;                 
ranks = zeros(nx, nBins);     

obs_steps = find(isObsStep);    
for var = 1:nx
    for idx = 1:numel(obs_steps)
        t = obs_steps(idx);                 
        j = obsIndexAtStep(t);            

        % Skip if missing observation
        if j <= 0 || j > size(y,2) || any(isnan(y(var,j))), continue; end

        % Ensemble at this time in history vs observation at time
        ens_vals = sort( squeeze(En_history(var, :, t)) );  
        obs_val  = y(var, j);                               

        % Rank: how many ensemble members are strictly less than obs
        r = sum(obs_val > ens_vals) + 1;
        ranks(var, r) = ranks(var, r) + 1;
    end
end

% Plot
figure('Name','Rank Histograms (Obs-based)','Color','w');
for var = 1:nx
    subplot(3,1,var);
    bar(1:nBins, ranks(var,:), 'FaceColor', [0.4 0.6 0.9]);
    xlim([1 nBins]);
    xlabel('Rank bin (1 .. Ne+1)');
    ylabel('Frequency');
    title(sprintf('Rank Histogram vs Observations — x_%d', var));
    grid on;
end

%% Ensemble forecast after window


figure();

% ---- x ----
subplot(3,1,1); hold on;
for m = 1:Ne
    plot(tgrid, squeeze(En_history(1,m,:)), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(tgrid, xtrue(1,:), 'k-', 'LineWidth', 2.0);
ylabel('x');
title('Ensemble Trajectories (Spaghetti) with Truth');
grid on;

% ---- y ----
subplot(3,1,2); hold on;
for m = 1:Ne
    plot(tgrid, squeeze(En_history(2,m,:)), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(tgrid, xtrue(2,:), 'k-', 'LineWidth', 2.0);
ylabel('y');
grid on;

% ---- z ----
subplot(3,1,3); hold on;
for m = 1:Ne
    plot(tgrid, squeeze(En_history(3,m,:)), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(tgrid, xtrue(3,:), 'k-', 'LineWidth', 2.0);
ylabel('z'); xlabel('Time');
grid on;

legend({'Ensemble Members','Truth'}, 'Location','best');

end
