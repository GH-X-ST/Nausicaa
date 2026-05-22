function C_D0 = CD0(Re, F, Q, S_wet, S_ref)
% Introduction:
%   Parasite drag coefficient for low speed flight in turbulent flow
%
% Author:
%   Hanchen Li (hl3422@ic.ac.uk)
%
% Inputs:
%   Re        - Reynolds number
%   F         - form factor
%   Q         - interference factor 
%   S_wet     - wetted surface area (m^2)
%   S_ref     - reference wing area (m^2)
%
% Outputs:
%   C_D0      - Parasite drag coefficient

%% 1 Calculate Skin-Friction Coefficient

C_f = 0.455 / ((log10(Re)) ^ 2.58);     % Prandtl–Schlichting formula

%% 2 Calculate Parasite Drag Coefficient

C_D0 = C_f * F * Q * (S_wet / S_ref);

end