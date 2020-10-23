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


%%
ia = 120;
Tstart = 0;
Tstop = 2400;
t = linspace(0,600,2400);
data = train_HR(1:Tstop,ia);

p1 = 0.999;
p2 = 0.001;
lambda1 = 1e5;
lambda2 = 1e6;
neg_removed_baseline = baseline(data,lambda1,p1);
pos_removed_baseline = baseline(neg_removed_baseline,lambda2,p2);
figure(1);hold on;
plot(t,data);
plot(t,neg_removed_baseline);
plot(t,pos_removed_baseline);

