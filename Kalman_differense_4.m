%% Очистка рабочего пространства
clear; clc; close all;

%% Параметры дискретизации
Fs = 1000;                      % Частота дискретизации, Гц
t = 0:1/Fs:2;                   % Время от 0 до 2 секунд
n = length(t);

%% Формирование исходного чистого сигнала (синус - трапеция - линейный тренд)
clean_signal = zeros(size(t));

% 1. Синусоида (0 – 0.6 с)
idx_sin = t <= 0.6;
f_sin = 5;                      % Частота синуса, Гц
clean_signal(idx_sin) = sin(2*pi*f_sin*t(idx_sin));

% 2. Трапеция (0.6 – 1.4 с)
idx_trap = (t > 0.6) & (t <= 1.4);
T_total = 1.4 - 0.6;            % 0.8 с
T_phase = T_total / 3;          % 0.2667 с – длительность нарастания и спада
% Опорные точки трапеции
t_break = [0.6, 0.6+T_phase, 0.6+2*T_phase, 1.4];
v_break = [0, 1, 1, 0];
clean_signal(idx_trap) = interp1(t_break, v_break, t(idx_trap), 'linear');

% 3. Линейный тренд (1.4 – 2.0 с)
idx_lin = t > 1.4;
clean_signal(idx_lin) = 2 * (t(idx_lin) - 1.4);   % от 0 до 1.2

%% Добавление переменного шума (для проверки адаптивного фильтра)
noise_power = 0.5 * ones(size(t));
noise_power((t > 0.5) & (t <= 1.5)) = 2.0;   % средний шум
noise_power(t > 1.5) = 4.0;                   % высокий шум

rng(42); % для воспроизводимости
noisy_signal = clean_signal + sqrt(noise_power) .* randn(size(clean_signal));

%% Определение параметров всех фильтров

% 1. Скалярный базовый
F_simple = 1;
H_simple = 1;
Q_simple = 0.05;
R_simple = 0.8;
x0_simple = noisy_signal(1);
P0_simple = 1;

% 2. Скалярный «корректно настроенный» (более инерционный)
Q_smooth_simple = 0.02;
R_smooth_simple = 1.5;

% 3. Векторный 2-го порядка (положение + скорость)
F = [1, 1/Fs; 0, 1];
H = [1, 0];
Q = [0.02, 0; 0, 5];          % большой Q для скорости – быстрая реакция
R = 0.5;
x0 = [noisy_signal(1); 0];
P0 = [1, 0; 0, 10];

% 4. Векторный 3-го порядка (положение + скорость + ускорение)
F3 = [1, 1/Fs, (1/Fs)^2/2; 0, 1, 1/Fs; 0, 0, 1];
H3 = [1, 0, 0];
Q3 = [0.01, 0, 0; 0, 0.1, 0; 0, 0, 0.5];   % ускорение может меняться
R3 = 2.0;                                    % сильное сглаживание
x03 = [noisy_signal(1); 0; 0];
P03 = [1, 0, 0; 0, 10, 0; 0, 0, 50];

% 5. Каскадный (быстрый + медленный)
R_fast = 0.3;
Q_fast = [0.05, 0; 0, 2];
R_slow = 5.0;
Q_slow = [0.005, 0; 0, 0.1];

% 6. Адаптивный (автоматическая подстройка R)
R_init = noise_power(1);   % начальное значение R

%% Запуск фильтрации

filtered_simple = filter_scalar(noisy_signal, F_simple, H_simple, Q_simple, R_simple, x0_simple, P0_simple);
filtered_smooth_simple = filter_scalar(noisy_signal, F_simple, H_simple, Q_smooth_simple, R_smooth_simple, x0_simple, P0_simple);
filtered_vector2 = filter_vector2(noisy_signal, F, H, Q, R, x0, P0);
filtered_vector3 = filter_vector3(noisy_signal, F3, H3, Q3, R3, x03, P03);

% Каскад: сначала быстрый, потом медленный (применяется к результату быстрого)
fast_stage = filter_vector2(noisy_signal, F, H, Q_fast, R_fast, x0, P0);
filtered_cascade = filter_vector2(fast_stage, F, H, Q_slow, R_slow, [fast_stage(1); 0], P0);

% Адаптивный
[filtered_adaptive, R_history] = filter_adaptive(noisy_signal, F, H, Q, x0, P0, R_init);

%% Сбор результатов для сравнения
signals = {filtered_simple, filtered_smooth_simple, filtered_vector2, ...
           filtered_vector3, filtered_cascade, filtered_adaptive};
names = {'Скалярный базовый', 'Скалярный инерционный', 'Векторный 2 порядка', ...
         'Векторный 3 порядка', 'Каскадный', 'Адаптивный'};

%% Вычисление MSE (относительно исходного чистого сигнала)
mse = zeros(length(signals), 1);
for i = 1:length(signals)
    mse(i) = mean((signals{i} - clean_signal).^2);
end

%% ========================================================================
%% ГРАФИК 1: исходный + зашумленный + все фильтры (фрагменты)
%% ========================================================================
figure('Position', [100, 100, 1400, 600]);

% График 1.1: исходный и зашумленный сигналы
subplot(2,3,1);
plot(t, clean_signal, 'k', 'LineWidth', 2);
hold on;
plot(t, noisy_signal, 'r', 'LineWidth', 0.5);
legend('Исходный', 'Зашумленный');
xlabel('Время, с'); ylabel('Амплитуда');
title('Исходный и зашумленный сигнал');
grid on; xlim([0 2]);

% График 1.2: все фильтры на одном поле
subplot(2,3,2);
plot(t, clean_signal, 'k', 'LineWidth', 2);
hold on;
colors = lines(length(signals));
for i = 1:length(signals)
    plot(t, signals{i}, 'Color', colors(i,:), 'LineWidth', 1.5);
end
legend(['Исходный', names], 'Location', 'eastoutside');
xlabel('Время, с'); ylabel('Амплитуда');
title('Сравнение всех фильтров');
grid on; xlim([0 2]);

% График 1.3: увеличенный фрагмент (синус)
subplot(2,3,3);
zoom_start = 0.2; zoom_end = 0.5;
idx = t>=zoom_start & t<=zoom_end;
plot(t(idx), clean_signal(idx), 'k', 'LineWidth', 2);
hold on;
for i = 1:length(signals)
    plot(t(idx), signals{i}(idx), 'Color', colors(i,:), 'LineWidth', 1.5);
end
legend(['Исходный', names], 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Фрагмент: синус');
grid on;

% График 1.4: увеличенный фрагмент (трапеция)
subplot(2,3,4);
zoom_start = 0.7; zoom_end = 1.3;
idx = t>=zoom_start & t<=zoom_end;
plot(t(idx), clean_signal(idx), 'k', 'LineWidth', 2);
hold on;
for i = 1:length(signals)
    plot(t(idx), signals{i}(idx), 'Color', colors(i,:), 'LineWidth', 1.5);
end
legend(['Исходный', names], 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Фрагмент: трапеция');
grid on;

% График 1.5: увеличенный фрагмент (линейный тренд)
subplot(2,3,5);
zoom_start = 1.5; zoom_end = 1.9;
idx = t>=zoom_start & t<=zoom_end;
plot(t(idx), clean_signal(idx), 'k', 'LineWidth', 2);
hold on;
for i = 1:length(signals)
    plot(t(idx), signals{i}(idx), 'Color', colors(i,:), 'LineWidth', 1.5);
end
legend(['Исходный', names], 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Фрагмент: линейный тренд');
grid on;

% График 1.6: столбчатая диаграмма MSE
subplot(2,3,6);
bar(1:length(mse), mse);
set(gca, 'XTick', 1:length(mse), 'XTickLabel', names, 'XTickLabelRotation', 45);
ylabel('MSE');
title('Среднеквадратичная ошибка');
grid on;

%% ========================================================================
%% ГРАФИК 2: групповое сравнение фильтров (отдельно по типам)
%% ========================================================================
figure('Position', [100, 100, 1400, 600]);

% 2.1 Скалярные фильтры
subplot(2,2,1);
plot(t, clean_signal, 'k', 'LineWidth', 2);
hold on;
plot(t, filtered_simple, 'b', 'LineWidth', 1.5);
plot(t, filtered_smooth_simple, 'm', 'LineWidth', 1.5);
legend('Исходный', 'Скалярный базовый', 'Скалярный инерционный', 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Скалярные фильтры');
grid on; xlim([0 2]);

% 2.2 Векторные фильтры
subplot(2,2,2);
plot(t, clean_signal, 'k', 'LineWidth', 2);
hold on;
plot(t, filtered_vector2, 'g', 'LineWidth', 1.5);
plot(t, filtered_vector3, 'c', 'LineWidth', 1.5);
legend('Исходный', 'Векторный 2 порядка', 'Векторный 3 порядка', 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Векторные фильтры');
grid on; xlim([0 2]);

% 2.3 Каскадный фильтр (с отображением быстрого этапа)
subplot(2,2,3);
plot(t, clean_signal, 'k', 'LineWidth', 2);
hold on;
plot(t, fast_stage, 'r--', 'LineWidth', 1.5);       % результат первого фильтра
plot(t, filtered_cascade, 'b', 'LineWidth', 1.5);   % итоговый каскад
legend('Исходный', 'Быстрый этап', 'Каскадный', 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Каскадный фильтр');
grid on; xlim([0 2]);

% 2.4 Адаптивный фильтр
subplot(2,2,4);
plot(t, clean_signal, 'k', 'LineWidth', 2);
hold on;
plot(t, filtered_adaptive, 'Color', [0.5 0 0.8], 'LineWidth', 1.5);
legend('Исходный', 'Адаптивный', 'Location', 'best');
xlabel('Время, с'); ylabel('Амплитуда');
title('Адаптивный фильтр');
grid on; xlim([0 2]);

%% Дополнительный график: адаптация R (для информации)
figure;
plot(t, noise_power, 'm', 'LineWidth', 2);
hold on;
plot(t, R_history, 'c--', 'LineWidth', 1.5);
legend('Истинная мощность шума', 'Адаптивная оценка R');
xlabel('Время, с'); ylabel('R');
title('Адаптация параметра R');
grid on; xlim([0 2]);

%% Вывод MSE в командное окно
fprintf('\n=== СРАВНЕНИЕ MSE ВСЕХ МЕТОДОВ ===\n');
fprintf('%-25s | MSE\n', 'Метод');
fprintf('%-25s-|------\n', repmat('-',1,25));
for i = 1:length(names)
    fprintf('%-25s | %.6f\n', names{i}, mse(i));
end

%% ------------------------------------------------------------------------
%% Вспомогательные функции 
%% ------------------------------------------------------------------------

function filtered = filter_scalar(signal, F, H, Q, R, x0, P0)
    n = length(signal);
    x = zeros(1, n);
    P = zeros(1, n);
    filtered = zeros(1, n);
    x(1) = x0;
    P(1) = P0;
    filtered(1) = x0;
    for k = 2:n
        x_pred = F * x(k-1);
        P_pred = F * P(k-1) * F + Q;
        K = P_pred * H / (H * P_pred * H + R);
        x(k) = x_pred + K * (signal(k) - H * x_pred);
        P(k) = (1 - K * H) * P_pred;
        filtered(k) = x(k);
    end
end

function filtered = filter_vector2(signal, F, H, Q, R, x0, P0)
    n = length(signal);
    x = zeros(2, n);
    P = zeros(2, 2, n);
    filtered = zeros(1, n);
    x(:,1) = x0;
    P(:,:,1) = P0;
    filtered(1) = x0(1);
    for k = 2:n
        x_pred = F * x(:,k-1);
        P_pred = F * P(:,:,k-1) * F' + Q;
        K = P_pred * H' / (H * P_pred * H' + R);
        x(:,k) = x_pred + K * (signal(k) - H * x_pred);
        P(:,:,k) = (eye(2) - K * H) * P_pred;
        filtered(k) = x(1,k);
    end
end

function filtered = filter_vector3(signal, F, H, Q, R, x0, P0)
    n = length(signal);
    x = zeros(3, n);
    P = zeros(3, 3, n);
    filtered = zeros(1, n);
    x(:,1) = x0;
    P(:,:,1) = P0;
    filtered(1) = x0(1);
    for k = 2:n
        x_pred = F * x(:,k-1);
        P_pred = F * P(:,:,k-1) * F' + Q;
        K = P_pred * H' / (H * P_pred * H' + R);
        x(:,k) = x_pred + K * (signal(k) - H * x_pred);
        P(:,:,k) = (eye(3) - K * H) * P_pred;
        filtered(k) = x(1,k);
    end
end

function [filtered, R_hist] = filter_adaptive(signal, F, H, Q, x0, P0, R_init)
    n = length(signal);
    x = zeros(2, n);
    P = zeros(2, 2, n);
    filtered = zeros(1, n);
    R_hist = zeros(1, n);
    x(:,1) = x0;
    P(:,:,1) = P0;
    filtered(1) = x0(1);
    R_hist(1) = R_init;
    
    innovation_history = zeros(10, 1);   % окно для оценки дисперсии
    
    for k = 2:n
        x_pred = F * x(:,k-1);
        P_pred = F * P(:,:,k-1) * F' + Q;
        
        innovation = signal(k) - H * x_pred;
        
        % Обновляем историю инноваций
        innovation_history = [innovation; innovation_history(1:end-1)];
        
        % Адаптация R
        if k > 10
            innov_var = var(innovation_history);
            R_hist(k) = 0.1 + 2 * innov_var;   % эмпирическая формула
        else
            R_hist(k) = R_hist(k-1);
        end
        
        K = P_pred * H' / (H * P_pred * H' + R_hist(k));
        x(:,k) = x_pred + K * innovation;
        P(:,:,k) = (eye(2) - K * H) * P_pred;
        filtered(k) = x(1,k);
    end
end