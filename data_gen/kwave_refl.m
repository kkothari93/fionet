%% Reflector imaging data generation
% This file is not to be run separately.
% It will be run via the qsub job.
%
% THIS FILE IS NOT MEANT TO BE RUN DIRECTLY.

clc;
close all;
addpath('/home/kkothar3/FIO/k-Wave/');

for i = n1:n2

% create the computational grid
Nx = 512;           % number of grid points in the x (row) direction
Ny = 512;           % number of grid points in the y (column) direction
dx = 2;        % grid point spacing in the x direction [m]
dy = 2;        % grid point spacing in the y direction [m]

kgrid = kWaveGrid(Nx, dx, Ny, dy);

MIN_SPEED = 2000;
MAX_SPEED = 3200;

img = loadImage(sprintf(strcat(img_dir, '%d.png'), i));
img = imresize(img, [Nx, Ny]);
img = 1 - img;

medium.sound_speed = img*(MAX_SPEED - MIN_SPEED) + MIN_SPEED;  % [m/s]
medium.sound_speed = repmat(medium.sound_speed(1:1, :), Nx, 1);
medium.density = 2650;

% define a single source point
source.p_mask = zeros(Nx, Ny);
source.p_mask(1, Ny/4) = 1;

% create time array
[t_array, t] = kgrid.makeTime(MAX_SPEED, 0.3, t_end);
t0 = 0.01;
f0 = 50;
source.p = -2.0*(t_array - t0).*(f0^2).*(exp(-(f0^2)*(t_array-t0).^2));
source.p = filterTimeSeries(kgrid, medium, source.p);

% define a sensor line on top
sensor.mask = zeros(Nx, Ny);
sensor.mask(1,:) = 1.0;
sensor.record = {'p', 'p_final'};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, 
	'PMLInside', false, 'RecordMovie', false);
p_direct = sensor_data.p;


% create the computational grid
Nx = 512;           % number of grid points in the x (row) direction
Ny = 512;           % number of grid points in the y (column) direction
dx = 2;        % grid point spacing in the x direction [m]
dy = 2;        % grid point spacing in the y direction [m]

kgrid = kWaveGrid(Nx, dx, Ny, dy);

MIN_SPEED = 2000;
MAX_SPEED = 3200;

img = loadImage(sprintf(strcat(img_dir, '%d.png'), i));
img = imresize(img, [Nx, Ny]);
img = 1 - img;

medium.sound_speed = img*(MAX_SPEED - MIN_SPEED) + MIN_SPEED;  % [m/s]
medium.density = 2650;

% define a single source point
source.p_mask = zeros(Nx, Ny);
source.p_mask(1, Ny/4) = 1;

% create time array
[t_array, t] = kgrid.makeTime(MAX_SPEED, 0.3, t_end);
t0 = 0.01;
f0 = 50;
source.p = -2.0*(t_array - t0).*(f0^2).*(exp(-(f0^2)*(t_array-t0).^2));
source.p = filterTimeSeries(kgrid, medium, source.p);

% define a sensor line on top
sensor.mask = zeros(Nx, Ny);
sensor.mask(1,:) = 1.0;

sensor.record = {'p', 'p_final'};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor,
	'PMLInside', false, 'RecordMovie', false);


p = sensor_data.p - p_direct;
p = imresize(p, [512,512]);

save(sprintf(strcat(save_dir, '%d_%d.mat'),i,0), 'p');

end
