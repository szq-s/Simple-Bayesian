clc
clear
close all
% 生成两个服从正态分布的数据集
% rng(1); % 设置随机种子，以确保结果可重复

mu1 = [1, 1]; % 第一个类别的均值
sigma1 = [1, 0.5; 0.5, 1]; % 第一个类别的协方差矩阵
X1 = mvnrnd(mu1, sigma1, 100); % 从第一个正态分布中生成100个样本

mu2 = [-1, -1]; % 第二个类别的均值
sigma2 = [1, -0.5; -0.5, 1]; % 第二个类别的协方差矩阵
X2 = mvnrnd(mu2, sigma2, 100); % 从第二个正态分布中生成100个样本

% 创建数据集和标签
X = [X1; X2];
Y = [ones(size(X1, 1), 1); 2 * ones(size(X2, 1), 1)];

% 随机打乱数据集顺序
idx = randperm(size(X, 1));
X = X(idx, :);
Y = Y(idx);

% 创建朴素贝叶斯分类器并训练模型
mdl = fitcnb(X, Y);

% 在训练集上进行预测
Y_pred = predict(mdl, X);

% 计算准确率
accuracy = sum(Y_pred == Y) / numel(Y);
disp(['准确率：', num2str(accuracy)]);

% 可视化数据集和分类结果
figure;

% 绘制第一个类别的数据点（蓝色）
scatter(X(Y==1, 1), X(Y==1, 2), 'b');
hold on;

% 绘制第二个类别的数据点（红色）
scatter(X(Y==2, 1), X(Y==2, 2), 'r');

% 绘制决策边界
x1range = linspace(min(X(:, 1)), max(X(:, 1)), 100);
x2range = linspace(min(X(:, 2)), max(X(:, 2)), 100);
[X1, X2] = meshgrid(x1range, x2range);
XGrid = [X1(:), X2(:)];
pred = predict(mdl, XGrid);
decisionmap = reshape(pred, size(X1));
contour(X1, X2, decisionmap);
title('分类结果');
legend('类别1', '类别2', '决策边界');
