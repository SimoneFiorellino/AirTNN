% File paths for the four CSV files
csvFilePaths = {'snrdb_airtnn_87195_grouped.csv', 'snrdb_airgnn_99563_grouped.csv', ...
                'snrdb_tnn_87195_grouped.csv', 'snrdb_gnn_99563_grouped.csv'};  % Replace with your actual CSV file paths

% Markers and colors for the plots
markers = {'o', 'o', 'o', 'o'};
colors = {'blue', 'red', 'green', 'magenta'};
names = {'airtnn', 'airgnn', 'tnn', 'gnn'};

% Create the plot with error bars
figure('Position', [0, 0, 2800, 2100]);  % [left, bottom, width, height]
hold on;

% Loop through each CSV file to plot data
for i = 1:length(csvFilePaths)
    % Read the table from the CSV file
    dataTable = readtable(csvFilePaths{i}, 'VariableNamingRule', 'preserve');
    
    % Extract columns from the table
    x_values = dataTable.('model/backbone/snr_db');
    y_values = dataTable.mean;
    y_std = dataTable.std;
    
    % Plotting for the current CSV file
    h = errorbar(x_values, y_values, y_std, markers{i}, 'MarkerSize', 6, 'MarkerEdgeColor', colors{i}, 'MarkerFaceColor', colors{i});
    set(h, 'HandleVisibility', 'off');  % Hide from legend
    plot(x_values, y_values, '-', 'Color', colors{i}, 'DisplayName', names{i});
end

% Add a flat curve with mean accuracy of 0.97
x = [0, 10, 20, 40];
y = repmat(0.97, size(x));
h = errorbar(x, y, NaN, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', 'cyan', 'MarkerFaceColor', 'cyan');
set(h, 'HandleVisibility', 'off');  % Hide from legend
plot(x, y, '-', 'Color', 'cyan', 'DisplayName', 'models without fading and noise');

hold off;  % Release the existing plot

% Add labels and title
xlabel('snr\_db');
xticks(0:10:40);
ylabel('Mean Accuracy');

% Add legend
legend('Location', 'east');

% Add grid
grid on;

% Remove extra white space by setting PaperPosition and PaperSize
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 10.667 8]);  % [left bottom width height] in inches
set(gcf, 'PaperSize', [10.667 8]);  % width x height in inches

% Save the plot
%set(gcf, 'PaperPositionMode', 'auto');
print('snrdb_plot', '-dpdf', '-r300');