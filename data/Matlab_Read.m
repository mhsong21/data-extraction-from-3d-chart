width = 0.5;

%% Matlab1
data1 = xlsread('Matlab_real_data','Matlab16','A1:AA100');
[numx,numy] = size(data1);
xs = [100 200 300 400 500 600];
ys = {'A', 'B', 'C'};
figure(1);
a = bar3(data1,width);
set(gca,'XTickLabel',xs);
set(gca,'YTickLabel',ys);

%% Matlab2
data2 = xlsread('Matlab_real_data','Matlab17','A1:AA100');
[numx,numy] = size(data2);
xs = [10 20 30 40 50 60 70];
ys = {'SA', 'FB', 'CF'};
figure(2);
b = bar3(data2, width);
set(gca,'XTickLabel',xs);
set(gca,'YTickLabel',ys);
for i = 1:1:numy
    b(i).FaceColor = [rand, rand, rand];
end

%% Matlab3
data3 = xlsread('Matlab_real_data','Matlab18','A1:AA100');
[numx,numy] = size(data3);
xs = [1 2 3 4 5 6 7];
ys = {1, 2, 3, 4};
figure(3);
c = bar3(data3, width);
set(gca,'XTickLabel',xs);
set(gca,'YTickLabel',ys);
for i = 1:1:numy
    c(i).FaceColor = [rand, rand, rand];
end

%% Matlab4
data4 = xlsread('Matlab_real_data','Matlab19','A1:AA100');
[numx,numy] = size(data4);
xs = {'A','B','C','D','E'};
ys = {10, 20, 30, 40, 50, 60, 70};
figure(4);
d = bar3(data4, width);
set(gca,'XTickLabel',xs);
set(gca,'YTickLabel',ys);
for i = 1:1:numy
    d(i).FaceColor = [rand, rand, rand];
end

%% Matlab5
data5 = xlsread('Matlab_real_data','Matlab20','A1:AA100');
[numx,numy] = size(data5);
xs = {'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug', 'Sep', 'Oct','Nov','Dec'};
ys = {'SA','SB','SC','SD','SE'};
figure(5);
e = bar3(data5, width);
set(gca,'XTickLabel',xs);
set(gca,'YTickLabel',ys);
%for i = 1:1:numy
%    e(i).FaceColor = [rand, rand, rand];
%end