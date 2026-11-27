function q3_detect_sharp_waves(data_dir, output_dir, mice)
% Detect sharp-wave candidates at each eligible center channel.
% Run in MATLAB with Signal Processing Toolbox:
%   addpath('code/questions'); q3_detect_sharp_waves

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin < 1, data_dir = 'E:/con2phys_submission/dataset_download/py-files'; end
if nargin < 2, output_dir = fullfile(repo, 'results', 'q3', 'psd_swr'); end
if nargin < 3, mice = 1:18; end
validateattributes(mice, {'numeric'}, {'vector', 'nonempty', 'integer', '>=', 1, '<=', 18});
assert(numel(unique(mice)) == numel(mice), 'Mouse IDs must be unique.');
mice = reshape(mice, 1, []);

areas = 1:3;
fs = 500;
detect_fs = 250;
band = [1 100];
filter_order = 3;
site_spacing_um = 20;
flank_distance_um = 100;
flank_rows = flank_distance_um / site_spacing_um;
threshold_sd = 3;
max_width_s = 0.100;
[z, p, k] = butter(filter_order, band / (fs / 2), 'bandpass');
[sos, gain] = zp2sos(z, p, k);

for mouse = mice
    for area = areas
        path = fullfile(data_dir, num2str(mouse), sprintf('lfp_%d.npy', area));
        assert(isfile(path), 'Missing recording: %s', path);
    end
end
if ~isfolder(output_dir), mkdir(output_dir); end
rate_path = fullfile(output_dir, 'q3_sharp_wave_channel_rates.csv');
event_path = fullfile(output_dir, 'q3_sharp_wave_events.csv');
[rates, msg] = fopen([rate_path '.partial'], 'w');
assert(rates >= 0, '%s', msg);
close_rates = onCleanup(@() fclose(rates));
[events, msg] = fopen([event_path '.partial'], 'w');
assert(events >= 0, '%s', msg);
close_events = onCleanup(@() fclose(events));

fprintf(rates, ['mouse_id,brain_area,center_row,deeper_row,superficial_row,status,' ...
    'duration_s,signal_sd_uv,threshold_uv,n_candidates,sharp_waves_per_min\n']);
fprintf(events, ['mouse_id,brain_area,center_row,peak_time_s,peak_height_uv,' ...
    'peak_height_sd,half_height_width_ms,prominence_uv\n']);

finished = 0;
for mouse = mice
    for area = areas
        path = fullfile(data_dir, num2str(mouse), sprintf('lfp_%d.npy', area));
        lfp = read_lfp(path);
        [n_samples, n_channels] = size(lfp);
        duration_s = n_samples / fs;
        valid_centers = 0;
        n_events = 0;
        for center = 1:n_channels
            deeper = center - flank_rows;
            superficial = center + flank_rows;
            if deeper < 1 || superficial > n_channels
                fprintf(rates, '%d,%d,%d,,,missing_flank,%.3f,,,,\n', ...
                    mouse, area, center, duration_s);
                continue
            end

            bipolar = lfp(:, superficial) - lfp(:, deeper);
            if any(~isfinite(bipolar)) || max(bipolar) == min(bipolar)
                fprintf(rates, '%d,%d,%d,%d,%d,invalid_signal,%.3f,,,,\n', ...
                    mouse, area, center, deeper, superficial, duration_s);
                continue
            end
            filtered = filtfilt(sos, gain, bipolar);
            trace = filtered(1 : fs / detect_fs : end) * 1e6;
            trace = trace - mean(trace);
            sigma = std(trace, 0);
            if ~isfinite(sigma) || sigma <= 0
                fprintf(rates, '%d,%d,%d,%d,%d,invalid_signal,%.3f,,,,\n', ...
                    mouse, area, center, deeper, superficial, duration_s);
                continue
            end
            threshold = threshold_sd * sigma;
            [height, time_s, width_s, prominence] = findpeaks(trace, detect_fs, ...
                'MinPeakHeight', threshold, 'MinPeakProminence', threshold / 2, ...
                'MaxPeakWidth', max_width_s, 'WidthReference', 'halfheight');
            count = numel(height);
            rate = count / (duration_s / 60);
            fprintf(rates, '%d,%d,%d,%d,%d,ok,%.3f,%.9g,%.9g,%d,%.9g\n', ...
                mouse, area, center, deeper, superficial, duration_s, sigma, threshold, count, rate);
            if count > 0
                rows = [repmat([mouse; area; center], 1, count); time_s(:)'; ...
                    height(:)'; height(:)' / sigma; width_s(:)' * 1000; prominence(:)'];
                fprintf(events, '%d,%d,%d,%.3f,%.9g,%.9g,%.9g,%.9g\n', rows);
            end
            valid_centers = valid_centers + 1;
            n_events = n_events + count;
        end
        clear lfp
        finished = finished + 1;
        fprintf('[%02d/%02d] Mouse %02d, area %d: %d/%d centers, %d channel detections\n', ...
            finished, numel(mice) * numel(areas), mouse, area, valid_centers, n_channels, n_events);
    end
end
clear close_rates close_events
[ok, msg] = movefile([rate_path '.partial'], rate_path, 'f');
assert(ok, '%s', msg);
[ok, msg] = movefile([event_path '.partial'], event_path, 'f');
assert(ok, '%s', msg);
gzip(event_path, output_dir);
delete(event_path);

config = struct('method_doi', '10.1038/s41467-024-44983-z', ...
    'event_label', 'sharp-wave candidate; ripple overlap not assessed', ...
    'mouse_ids', mice, 'brain_areas', areas, 'input_fs_hz', fs, 'detection_fs_hz', detect_fs, ...
    'filter_hz', band, 'filter_order', filter_order, 'filter', 'zero-phase Butterworth', ...
    'channel_spacing_um', site_spacing_um, 'flank_distance_um', flank_distance_um, ...
    'flank_offset_rows', flank_rows, 'row_order', 'one-based, deepest to most superficial', ...
    'bipolar_trace', 'more superficial flank minus deeper flank', ...
    'threshold_sd', threshold_sd, 'threshold_baseline', 'mean-centered bipolar trace', ...
    'min_prominence_sd', threshold_sd / 2, 'max_width_ms', max_width_s * 1000, ...
    'width_reference', 'halfheight', 'rate_denominator', 'full recording duration in minutes');
[meta, msg] = fopen(fullfile(output_dir, 'q3_sharp_wave_config.json'), 'w');
assert(meta >= 0, '%s', msg);
close_meta = onCleanup(@() fclose(meta));
fprintf(meta, '%s\n', jsonencode(config, 'PrettyPrint', true));
end


function lfp = read_lfp(path)
% C-order NumPy data [channel, time] load directly as MATLAB [time, channel].
[input, msg] = fopen(path, 'r', 'ieee-le');
assert(input >= 0, '%s', msg);
close_input = onCleanup(@() fclose(input));
magic = fread(input, 6, 'uint8')';
assert(isequal(magic, [147 double('NUMPY')]), 'Invalid NumPy file: %s', path);
version = fread(input, 2, 'uint8')';
switch version(1)
    case 1
        header_length = fread(input, 1, 'uint16');
    case {2, 3}
        header_length = fread(input, 1, 'uint32');
    otherwise
        error('Unsupported NumPy version in %s.', path);
end
header = char(fread(input, header_length, 'uint8')');
shape = regexp(header, '\(\s*(\d+)\s*,\s*(\d+)\s*\)', 'tokens', 'once');
assert(~isempty(shape) && contains(header, '''descr'': ''<f8''') && ...
    contains(header, '''fortran_order'': False'), ...
    'Expected a C-order little-endian float64 channel-by-time array: %s', path);
channels = str2double(shape{1});
samples = str2double(shape{2});
assert(channels > 0 && samples > 50, 'Invalid LFP dimensions in %s.', path);
lfp = fread(input, [samples channels], '*double');
assert(numel(lfp) == samples * channels, 'Truncated NumPy payload in %s.', path);
end
