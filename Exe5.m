clear all; 
close all; 
clc;

%Importação dos dados
entrada = (importdata('LDR_ball.txt'))';
%Criando os vetores de entrada e saída para o sensor inferencial regressor 
v1 = entrada(1,:); 
v2 = entrada(2,:);
v = [v1;v2];
y = entrada(3,:);

%Visualização dos dados de entrada (vensão dos LDRs em função da posição do
%objeto)
figure,
plot(y,v1,'b*',y,v2,'r*'),title(['Medição de tensão em função da posição do objeto'])
legend('V1','V2');

%Normalização dos dados
[vn1,vs1] = mapminmax(v1); 
[vn2,vs2] = mapminmax(v2);
vn = [vn1;vn2]; %Vetor de entrada normalizado
[yn,ys] = mapminmax(y); %vetor de saída normalizado

%Visualização dos dados de entrada
figure,
subplot(221),plot(v1,y,'b*'),title('V1')
subplot(223),plot(vn1,yn,'b*'),title('V1 Normalizado')
subplot(222),plot(v2,y,'b*'),title('V2')
subplot(224),plot(vn2,yn,'b*'),title('V2 Normalizado')


% Create a Fitting Network
trainFcn = 'trainlm';  % Algoritmo de Levenberg-Marquardt
hiddenLayerSize = 5; %  camadas oculta
net = fitnet(hiddenLayerSize,trainFcn);
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

% Train the Network
[net,tr] = train(net,v,y);

% Test the Network
y_1 = net(v);
E = rmse(y,y_1) 
% e = mean(abs(gsubtract(t,y)));%calcula o erro absoluto medio(MAE) entre t e y
% 
% figure
% plot(v1,y,'r*',v1,y_1,'b*'),
% legend('referencia','estimado M1');
% title('Modelo 1 camada oculta');
% 
% figure
% plot(v2,y,'r*',v2,y_1,'b*');
% legend('referencia','estimado M2');
% title('Modelo 1 camada oculta');

figure,
subplot(211),plot(v1,y,'r*',v1,y_1,'b*')
title('Modelo 5 camadas ocultas - V1')
legend('referencia','Estimado');
subplot(212),plot(v2,y,'r*',v2,y_1,'b*');
title('Modelo 5 camadas ocultas - V2')
legend('referencia','Estimado');
