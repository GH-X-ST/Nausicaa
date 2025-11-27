function [CL, CDi, y, Gamma, c_values, alpha_i, iteration] = ...
    LLT(U_inf, c_0, lambda, b, a0, AoA, N)
% Introduction:
%   Numerical solution of the lifting-line equation for a linearly tapered,
%   unswept wing, based on Fundamentals of Aerodynamics (Anderson)
%
% Author:
%   Hanchen Li (hl3422@ic.ac.uk)
%
% Inputs:
%   U_inf     - free stream velocity (m/s)
%   c_0       - root chord length (m)
%   lambda    - taper ratio
%   b         - span of the wing (m)
%   a0        - lift curve slope (per radian)
%   AoA       - angle of attack (rad)
%   N         - number of wing section
%
% Outputs:
%   CL        - Lift coefficient
%   CDi       - Induced drag coefficient
%   y         - spanwise station locations
%   Gamma     - spanwise circulation distribution
%   c_values  - spanwise chord length
%   alpha_i   - spanwise induced angle of attack distribution
%   iteration - number of iterations to compute result

%% 0 Basic Parameters

% 0.1 Basic parameters
s = b / 2;               % half-span of the wing (m)
c = @(y) c_0 * (1 - (1 - lambda)...
    * abs(y / s));       % sectional chord length
alpha = @(y) AoA;        % sectional twist angle

% 0.2 Discretize the wing 
k = N + 1;               % number of spanwise stations
dy = b / N;              % step size for y
y = linspace(-s, s, k);  % spanwise station locations

% 0.3 Find chord length
c_values = c(y);         % chord length at all station

if mod(k, 2) == 0        % number of intervals must be odd

    N = N + 1;
    k = N + 1;
    y = linspace(-s, s, k);
    dy = b / N;
    c_values = c(y);

end

% 0.4 Calculate average chord length using Simpson's rule
c_avg = (dy / 3) * ...
    (c_values(1) + ...
    4 * sum(c_values(2 : 2 : end - 1)) + ...
    2 * sum(c_values(3 : 2 : end - 2)) + ...
    c_values(end)) / (2 * s);

% 0.5 Wing parameters
AR = b / c_avg;           % aspect ratio
S_ref = b * c_avg;        % reference area

% 0.6 Initial guess
Gamma = 0.1 * ones(1, k); % Initial guess for circulation distribution
alpha_i = zeros(1, k);    % Initial guess for induced angle of attack

% 0.7 Set boundary condition
Gamma(1) = 0;             % circulation at the wing tip (y = -s)
Gamma(k) = 0;             % circulation at the wing tip (y = s)

%% 1 Iterative Solver

% 1.1 Basic parameters
D = 0.01;                 % learning rate
tolerance = 1e-5;         % convergence tolerance
error = inf;              % pre-define error
iteration = 0;            % number of iteration

% 1.2 find the circulation distribution

while error > tolerance

    Gamma_old = Gamma;    % define guessed circulation

    % compute induced angle of attack (alpha_i) at each station

    for n = 1 : k
        
        integral = 0;     % Pre-define integral

        for j = 1 : k

            if j ~= n

                % compute dGamma/dy using central differencing scheme

                if j == 1

                    dGammady = (Gamma(j + 1) - Gamma(j)) / dy;

                elseif j == k

                    dGammady = (Gamma(j) - Gamma(j - 1)) / dy;

                else

                    dGammady = (Gamma(j + 1) - Gamma(j - 1)) / (2 * dy);

                end

                % handle singularity by averaging adjacent sections

                if abs(y(n) - y(j)) < 1e-6

                    if j == 1

                        integral = integral + dGammady / ((y(n)...
                            - y(j + 1)) / 2);

                    elseif j == k

                        integral = integral + dGammady / ((y(n)...
                            - y(j - 1)) / 2);

                    else

                        integral = integral + dGammady / ((y(n)...
                            - y(j - 1)) / 2 + (y(n) - y(j + 1)) / 2);

                    end

                else

                    integral = integral + dGammady / (y(n) - y(j));

                end

            end

        end

        alpha_i(n) = (1 / (4 * pi * U_inf)) * dy * integral;

    end

    % calculate effective angle of attack
    alpha_e = alpha(y) - alpha_i;

    % update circulation distribution using sectional lift coefficient
    for n = 1:k

        c_n = c(y(n));

        CL_n = a0 * alpha_e(n);

        Gamma(n) = 0.5 * U_inf * c_n * CL_n;

    end

    % apply boundary conditions again
    Gamma(1) = 0;     % circulation at the wing tip (y = -s)
    Gamma(k) = 0;     % circulation at the wing tip (y = s)

    % iteration
    Gamma = Gamma_old + D * (Gamma - Gamma_old);

    % calculate error for convergence
    error = max(abs(Gamma - Gamma_old));
    iteration = iteration + 1;

end

%% 2 Lift Coefficient

% Simpson's rule
CL = (2 / (U_inf * S_ref)) * (dy / 3) * ...
    (Gamma(1) + ...
    4 * sum(Gamma(2 : 2 : end - 1)) + ...
    2 * sum(Gamma(3 : 2 : end - 2)) + ...
    Gamma(end));

%% 3 Induced Drag Coefficient

% Simpson's rule
CDi = (2 / (U_inf * S_ref)) * (dy / 3) * ...
    (Gamma(1) * alpha_i(1) + ...
    4 * sum(Gamma(2 : 2 : end - 1) .* alpha_i(2 : 2 : end - 1)) + ...
    2 * sum(Gamma(3 : 2 : end - 2) .* alpha_i(3 : 2 : end - 2)) + ...
    Gamma(end) * alpha_i(end));

end