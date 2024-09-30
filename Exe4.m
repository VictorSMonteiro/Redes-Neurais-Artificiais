close all; clear all; clc;

dados_treinamento = (importdata('treinamento2.dat'))';
dados_teste = (importdata('teste2.dat'))';

%Criando os vetores de entrada e saída para treinamento e teste da rede
x = dados_treinamento(1:4,:); %vetor com os dados de treinamento (entrada)
t = dados_treinamento(5,:); %vetor com os dados de treinamento (saída)
x_t = dados_teste(1:4,:); %vetor com os dados de teste (entrada)
num_treino = size(x,2); %Qtd amostras de treino
num_teste = size(x_t,2); %Qtd amostras de teste
num_entrada = size(x,1); %Qtd de entradas

% Normalização dos dados de treinamento
[xn, xs] = mapminmax(x);
[tn, ts] = mapminmax(t);
% Normalização dos dados de teste
[x_t_n, x_t_s] = mapminmax(x_t);

%Iniciando o vetor w com valores aleatórios pequenos
w = rand(1, num_entrada); % vetor de pesos 
bias = rand; % bias 
taxa_aprend = 0.5; % Especificando a taxa de aprendizado
precisao = 0.001; % Especificando a precisão requerida

num_epocas = 300; % Número de épocas de treinamento
Erro = zeros(1,num_epocas); %Criando vetor de erro
dif = 100;
W_inicial = w;
bias_inicial = bias;

Eqm = zeros(num_epocas,1); %Criando vetor do erro quadrático médio

j=2;
while (j < num_epocas & dif > precisao)
     for i = 1:num_entrada
        x_i = xn(:, i);
        d_i = t(i);
        u_i = (w * x_i) + bias;
        e_i = d_i - u_i;
        w = w + (taxa_aprend * e_i * x_i)'; %Regra delta
        bias = bias + taxa_aprend * e_i;
        Eqm(j) = Eqm(j) + (e_i*e_i);
    end
    Eqm(j) = Eqm(j)/num_entrada;
    dif = abs(Eqm(j) - Eqm(j-1));
    j=j+1;
end


 for i = 1:num_treino
        x_i = xn(:, i);
        y_pred(i) = (w * x_i) + bias;
        y_pred2(i)=mapminmax('reverse',y_pred(i),ts);
        if y_pred2(i) <= 0
            y_2(i) = -1;
        else
            y_2(i) = 1;
        end;
 end

valvula = string(num_teste);
for i=1:num_teste
         x_t_i = x_t_n(:,i);
        y_teste(i) = (w * x_t_i) + bias;
        y_teste2(i)=mapminmax('reverse',y_teste(i),ts);
        if y_teste2(i) <= 0
            y_3(i) = -1;
            valvula(i) = 'A';
        else
            y_3(i) = 1;
            valvula(i) = 'B';
        end;
end

disp(valvula);

figure,
plot(Eqm(1:j))

