% Load the .mat file
V0_mat = load('V_EIM0.mat');

% Assuming the variable name in the .mat file is 'V0' (adjust this if the actual variable name is different)
V0 = V0_mat.V_EIM0;

% Print the first element
disp(V0(1, 1, 4));
