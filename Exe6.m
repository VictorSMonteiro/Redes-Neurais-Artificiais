close all; clear all; clc;

carga_eletrica = (importdata('carga_eletrica.txt'))';
figure, plot (carga_eletrica);
title('Potência Ativa');

figure, boxplot(carga_eletrica);
title('Boxplot dados de potência ativa');

%Dividindo os dados da série temporal treinamento/validação/testes do 
% previsor e comparação com a previsão realizada
n_total = length(carga_eletrica); %número total de amostras
n_comp = 24; % número de amostras para comparação com os valores previstos
% (previssões recursivas de 24 passos à frente)
n_treino = n_total - n_comp; %número de amostras treino/teste/validação

potencia = carga_eletrica(1:n_treino); %amostras treino/teste/validação
potencia_medida = carga_eletrica(n_treino+1:n_total); %amostras para 
% comparação com os valores previstos 

%Criando os vetores de entrada e saída para previsão da série temporal
for i=1:(n_treino-24)
    t(1,i)=potencia(24+i);% Medição de potência no instante k -> PA(k)
    x(1,i)=potencia(23+i);% Medição de potência no instante k-1 -> PA(k-1)
    x(2,i)=potencia(22+i);% Medição de potência no instante k-2 -> PA(k-2)
    x(3,i)=potencia(i); % Medição de potência no instante k-24 -> PA(k-24)
end;

%Código abaixo gerado a partir da toolbox nftool
% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
num_layer = 28;
hiddenLayerSize = num_layer;
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
performance= perform(net,t,y);

% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, plotregression(t,y)

z1=1:length(t);
figure, plot(z1,y(1,:),'b-',z1,t(1,:),'r-')
title('Comparação valores medidos e estimados');
legend('estimado','medido');

%*************************************************
% Previsões recursivas de 1 a 24 passos à frente
%*************************************************
num_passos = 24;
z = 1:num_passos;

for i=1:num_passos
    dados(1,i) = potencia(end);
    dados(2,i) = potencia(end-1);
    dados(3,i) = potencia(end-24);
    y_ch(i) = net(dados(:,i));

    potencia = [potencia y_ch(i)];
end;

figure,plot(z,y_ch','b-',z,potencia_medida,'r-')
title('Comparação dos valores previstos e medidos');
legend('previsto','medido');

E = mape(y_ch,potencia_medida)

