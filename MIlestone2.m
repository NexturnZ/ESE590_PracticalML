df1 = load('./data/FHR/FHRDataCol.csv');
df2 = load('./data/FHR/UADataCol.csv');

%%
ia = 7;
Tstart = 0;
Tstop = 2400;
data = df1(1:Tstop,ia);

p = 0.9;
lambda = 1e4;
z = baseline(data,lambda,p);
figure(1);hold on;
plot(data);
plot(z);

