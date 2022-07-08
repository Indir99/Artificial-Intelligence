clear all
close all
clc

% Plot za izgled signala:
%x=0:0.01:10; 
%x=x';
%y=(cos(x)).^2 + 3*sin(3*x);
%plot(y)
%grid


%Kod sa vjezbi
x=0:0.05:10; 
x=x';
y=(cos(x)).^2 + 3*sin(3*x);
plot(y)
grid
numPts=size(x(:,1));
data =[x y];
trndata=data(1:2:numPts,:);
chkdata=data(2:2:numPts,:);
plot(trndata(:,1),trndata(:,2),'o',chkdata(:,1),chkdata(:,2),'x')
grid
title('Trenirajuci podaci (o) i podaci za provjeru (x)')
xlabel('x');
ylabel('Podaci za y');
hold on


% Dodatni kod:
% gaussmf, 15 mfs, 100 epoch
options = genfisOptions('GridPartition','OutputMembershipFunctionType','constant','InputMembershipFunctionType','gaussmf','NumMembershipFunctions',15);
fismat = genfis(x,y,options);
%figure(2)
%plotmf(fismat,'input',1)

options = anfisOptions;
options.InitialFIS = fismat;
options.EpochNumber = 100;
options.ValidationData = chkdata;
[fismat1, trnerr, ss, fismat2, chkerr] = anfis(trndata, options);
figure(2)
plotmf(fismat1,'input',1)

figure(3)
out = evalfis(chkdata(:,1),fismat1);
hold;
plot(chkdata(:,1),out,'b');
hold on;
plot(chkdata(:,1),chkdata(:,2));
plot(x,y,'r');
grid

