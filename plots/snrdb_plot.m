% File paths for the four CSV files
csvFilePaths = {'snrdb_airtnn_grouped.csv', 'snrdb_airgnn_grouped.csv', ...
                'snrdb_tnn_grouped.csv', 'snrdb_gnn_grouped.csv'};  % Replace with your actual CSV file paths

set(0, 'defaultTextInterpreter', 'latex');
set(0, 'defaultAxesTickLabelInterpreter', 'latex');
set(0, 'defaultLegendInterpreter', 'latex');
set(0, 'defaultAxesFontName', 'Times New Roman');

% Markers and colors for the plots
markers = {'-o', '-x', '-diamond', '->'};
colors = {'#0072BD', '#D95319', '#EDB120', '#77AC30'};
names = {'AirTNN', 'AirGNN', 'TNN', 'GNN'};

% Create the plot with error bars
figure('Position', [0, 0, 600, 450]);  % [left, bottom, width, height]
hold on;

% Loop through each CSV file to plot data
for i = 1:length(csvFilePaths)
    % Read the table from the CSV file
    dataTable = readtable(csvFilePaths{i}, 'VariableNamingRule', 'preserve');
    
    % Extract columns from the table
    x_values = dataTable.('model/backbone/snr_db');
    y_values = dataTable.mean;
    
    % Plotting for the current CSV file
    %h = errorbar(x_values, y_values, NaN, markers{i}, 'MarkerSize', 6, 'MarkerEdgeColor', colors{i}, 'LineWidth', 3);
    %set(h, 'HandleVisibility', 'off');  % Hide from legend
    plot(x_values, y_values, markers{i}, 'MarkerSize', 12, 'Color', colors{i}, 'DisplayName', names{i}, 'LineWidth', 3);
end

% Add a flat curve with mean accuracy of 0.97
%x = [0.25, 0.5, 1, 1.5, 2];
%y = repmat(0.96, size(x));
%h = errorbar(x, y, NaN, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', 'cyan', 'MarkerFaceColor', 'cyan');
%set(h, 'HandleVisibility', 'off');  % Hide from legend
%plot(x, y, '-', 'Color', 'cyan', 'DisplayName', 'models without fading and noise');

yline(0.96, '--', 'Color', [0, 0, 0], 'LineWidth', 3, 'DisplayName', 'Ideal setting');

hold off;  % Release the existing plot

% Set font size for axes tick labels
ax = gca;
ax.FontSize = 16;  % Set font size for axes
ax.LineWidth = 1;

% Add labels and title
xticks(0:5:40);
xlim([0, 40]);

xlabel('$SNR\ (dB)$', 'Interpreter', 'latex', 'FontSize', 21);
ylabel('Accuracy', 'Interpreter', 'latex', 'FontSize', 21);


% Add legend
legend('Location', 'east', 'FontSize', 21, 'LineWidth', 0.01);

% Add grid
grid on;

% Remove extra white space by setting PaperPosition and PaperSize
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 10.667 8]);  % [left bottom width height] in inches
set(gcf, 'PaperSize', [10.667 8]);  % width x height in inches

% Save the plot
% set(gcf, 'PaperPositionMode', 'auto');
print('snr_plot', '-dpdf');