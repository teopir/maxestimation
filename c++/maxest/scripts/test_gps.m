clc;
clear all
rng(0,'twister'); % For reproducibility

N = 50;
x = 2 * ones(N,1);
y = randn(size(x));
y_mean = mean(y)
y_std = std(y)
y_mean_std = y_std / sqrt(length(x))
plot(x,y,'o');

gprMdl = fitrgp(x,y,'Basis','linear', 'KernelFunction', 'squaredexponential',...
      'FitMethod','exact','PredictMethod','exact');
  
[ypred,ysd] = predict(gprMdl,2);

fprintf('\n[matgp] pred: %f, std: %f\n', ypred(end), ysd(end))

dlmwrite('/home/matteo/Projects/maxest/c++/build-maxest-Desktop-Default/examples/x.dat', x)
dlmwrite('/home/matteo/Projects/maxest/c++/build-maxest-Desktop-Default/examples/y.dat', y)

format long

sf_val = gprMdl.KernelInformation.KernelParameters(2);
sn_val = gprMdl.Sigma;
l_val = gprMdl.KernelInformation.KernelParameters(1);

fprintf('x.dat y.dat %.18f %.18f %.18f\n', l_val, sf_val, sn_val);


%%
sf_val = gprMdl.KernelInformation.KernelParameters(2);
sn_val = gprMdl.Sigma;
l_val = gprMdl.KernelInformation.KernelParameters(1);
k = @(v,z) sf_val^2 * exp((v-z)^2 / (2*l_val^2));

xstar = 2;

K = zeros(N,N);
KXXS = zeros(N,1);
for i = 1:N
    KXXS(i) = k(xstar, x(i));
    for j=1:N
        K(i,j) = k(x(i),x(j));
    end
end

T = K + eye(N) * sn_val^2;
var = k(xstar, xstar) - KXXS'*(T\KXXS) + sn_val^2;
pred = KXXS' * (T\y);

fprintf('\npred: %f, std: %f\n', pred, sqrt(var))


%%
syms sf sn l
val = sf^2 * exp(-1.0 / (2*l^2));
K = ones(N,N) * sf^2;
KXXS = ones(N,1)*sf^2;
T = K + eye(N) * sn^2;
pred = transpose(KXXS) * inv(T) * y;
var = simplify(sf^2  - transpose(KXXS) * inv(T) * KXXS + sn^2);
var

sf = gprMdl.KernelInformation.KernelParameters(2);
sn = gprMdl.Sigma;
l = gprMdl.KernelInformation.KernelParameters(1);

eval(subs(transpose(KXXS) * inv(T)))
fprintf('\n[symb] pred: %f, std: %f\n', eval(subs(pred)), sqrt(eval(subs(var))))
