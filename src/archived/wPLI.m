%% OTKA_wPLI
directory = '/Users/yeganeh/Documents/MATLAB/DATA4MATLAB_PLI';
cd(directory);
sub_files = dir('*.fif');
nsubs = size(sub_files, 1);

h = waitbar(0,'Running Connectivity Analysis');

%initiate a container to collect connectivities
all_conns = [];

for i = 1:nsubs

     waitbar(i/nsubs);

    %% Open data
    filename = sub_files(i).name;
    disp(strcat('Working on file ', filename));

    cfg = [];
    cfg.dataset = filename;
    fif_data = ft_preprocessing(cfg);

    %% Epoching
    cfg_epoching = [];
    cfg_epoching.length  = 2;
    cfg_epoching.overlap = 0;
    epochs = ft_redefinetrial(cfg_epoching, fif_data);


    %% Frequency analysis
    cfg_freq = [];
    cfg_freq.method = 'mtmfft';
    cfg_freq.output = 'powandcsd';
    cfg_freq.channel = 1:12;
    cfg_freq.keeptrials ='yes'; %do not return an average of all trials for subsequent wpli analysis
    cfg_freq.taper = 'hanning';

    %alpha2
    cfg_freq.foi         = 10.5:0.5:12;
    freq_alpha2 = ft_freqanalysis(cfg_freq, epochs);

    %% Connectivity
    cfg_conn = [];
    cfg_conn.method = 'wpli';
    conn_alpha2 = ft_connectivityanalysis(cfg_conn, freq_alpha2);

    %% append to all_conns using an apt name
    name = split(filename, '-');
    name = split(name(2), '_');
    name = char(append(name(2), '_', name(1)));
    all_conns.(name) = conn_alpha2.wplispctrm;   

end

%Append other connectivity fields into the conn_alls
conn_fields = fieldnames(conn_alpha2);

for k=1:length(conn_fields)
    %add every other field except this wpli spectrum
    if ~strcmp(conn_fields{k}, 'wplispctrm')
        all_conns.info.(conn_fields{k}) = conn_alpha2.(conn_fields{k});
    end
end

%change directory to save the mat file
directory = '/Users/yeganeh/Documents/MATLAB';
cd(directory);
save('all_wpli.mat', 'all_conns');