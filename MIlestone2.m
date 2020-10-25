clear;
clc;
%% Data loading
HR = load('./data/FHR/FHRDataCol.csv'); % heart rate, size: 8000*552(PointsPerSample*Samples)
UC = load('./data/FHR/UADataCol.csv'); % Uterus compression, size: 8000*552(PointsPerSample*Samples)

num_samples = size(HR,2);
train_HR = HR(:,1:num_samples/3);
validation_HR = HR(:,num_samples/3+1:2*num_samples/3);
test_HR = HR(:,2*num_samples/3+1:end);

train_UC = UC(:,1:num_samples/3);
validation_UC = UC(:,num_samples/3+1:2*num_samples/3);
test_UC = UC(:,2*num_samples/3+1:end);


%% Baseline
ia = 24; %120, 60, 77, 90, 38, 24
Tstart = 0;
Tstop = 2400;
slot_len = 800; % length of each time slot
data = train_HR(1:Tstop,ia);
t = linspace(0,length(data)/4,length(data));

p1 = 0.999;
p2 = 0.001;
lambda1 = 1e4;
lambda2 = 1e7;
neg_removed_baseline = zeros(length(data),1);
pos_removed_baseline = zeros(length(data),1);
bl = zeros(length(data),1);
for i1 = 1:length(data)/slot_len
    temp_neg_ = baseline(data((i1-1)*slot_len+1:i1*slot_len),lambda1,p1);
    temp_neg = baseline(temp_neg_,lambda2,p2);
    neg_removed_baseline((i1-1)*slot_len+1:i1*slot_len) = temp_neg;
    
%     temp_pos_ = baseline(data((i1-1)*slot_len+1:i1*slot_len),lambda2,p2);
%     temp_pos = baseline(temp_pos_,lambda1,p1);
%     pos_removed_baseline((i1-1)*slot_len+1:i1*slot_len) = temp_pos;
    
    bl((i1-1)*slot_len+1:i1*slot_len) = mean(temp_neg);
    
end


% neg_removed_baseline_ = baseline(data,lambda1,p1);
% neg_removed_baseline = baseline(neg_removed_baseline_,lambda2,p2);
% 
% pos_removed_baseline_ = baseline(data,lambda2,p2);
% pos_removed_baseline = baseline(pos_removed_baseline_,lambda1,p1);
% bl = (neg_removed_baseline+pos_removed_baseline)/2;

figure(1);hold on;grid on;
plot(t,data);
% plot(t,neg_removed_baseline);
plot(t,bl,'LineWidth',2);
title('Baseline');

%% Variability
std = data-bl;
win = 30*4;
vari = zeros(1,length(data));
for i1 = 1:floor(length(data)/win)
    vari((i1-1)*win+1:i1*win) = max(std((i1-1)*win+1:i1*win))-min(std((i1-1)*win+1:i1*win));
end

figure(2);
hold on;grid on;
plot(t,vari);
plot(t,6*ones(1,length(t)));
plot(t,25*ones(1,length(t)));

%% Acceleration
rol_mean = movmean(data,60); % rolling mean with window size of 60
[peak_val,peak_loc] = findpeaks(rol_mean,'MinPeakProminence',5);

% filter peaks that are lower than 15 bpm over baselines
temp = peak_val(peak_val>bl(peak_loc)+15);
peak_loc = peak_loc(peak_val>bl(peak_loc)+15);
peak_val = temp;


acc = zeros(length(peak_loc),2);
for i1 = 1:length(peak_loc)
    [peak_start_tmp,peak_stop_tmp] = len_peak(data,peak_loc(i1),bl);
    acc(i1,:) = [peak_start_tmp,peak_stop_tmp];
end
acc = acc(acc(:,2)-acc(:,1)>=60,:);
acc_loc = peak_loc(acc(:,2)-acc(:,1)>=60,:);

figure;
hold on; grid on;
plot(t, data);
plot(t, bl,'LineWidth',2);
plot(t,bl+15);
for i1 = 1:length(acc_loc)
   plot(t(acc(i1,1):acc(i1,2)),data(acc(i1,1):acc(i1,2)),'r','LineWidth',1); 
end



