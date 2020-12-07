%% Boundary source problem data generation
% This file is not to be run separately.
% It will be run via the qsub job.
%
% THIS FILE IS NOT MEANT TO BE RUN DIRECTLY.

clc;
close all;
addpath('/home/kkothar3/FIO/k-Wave/');

for i = n1:n2

% create the computational grid
Nx = 512;      % number of grid points in the x (row) direction
Ny = 512;      % number of grid points in the y (column) direction
dx = 2;        % grid point spacing in the x direction [m]
dy = 2;        % grid point spacing in the y direction [m]

kgrid = kWaveGrid(Nx, dx, Ny, dy);
MIN_SPEED = 1400;
MAX_SPEED = 4000;

% define the properties of the propagation medium
z = ones(512, 512);
z(200,220) = -31000;
imf = imgaussfilt(z, 200);
imf = imf - min(min(imf)); imf = imf/max(max(imf));

medium.sound_speed = imf*(MAX_SPEED - MIN_SPEED) + MIN_SPEED;  % [m/s]
medium.density = 2650;

% create time array
[t_array, t] = kgrid.makeTime(medium.sound_speed, 0.3, t_end);

% load mnist data
img = loadImage(sprintf(strcat(img_dir, '%d.png'), i));
img = imresize(img, [Nx/2,Ny/2]);
img = 1-img;

% rotate image randomly
g = zeros(Nx, Ny);
g(1:Nx/2, Ny/4+1:3*Ny/4) = img;
source.p0 = g;

% define a sensor line on top
sensor.mask = zeros(Nx, Ny);
sensor.mask(1,:) = 1.0;

sensor.record = {'p', 'p_final'};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
    'PMLInside', false, 'RecordMovie', false);

% store the data
p = sensor_data.p;
p_f = sensor_data.p_final;

p = imresize(p, [128,128]);
p_final = imresize(p_f, [128,128]);

save(sprintf(strcat(save_dir, '%d_%d.mat'),i,0), 'p', 'p_final');

end
