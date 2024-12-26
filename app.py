import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

def check_nelson_rules(values, mean, sigma):
    # print(values)
    # print(mean)
    # print(sigma)
    n = len(values)
    rules_violations = {i: [] for i in range(1, 9)}
    checked_points = set()  # Keep track of points that have already violated a rule

    # Rule 1: Points > 3 sigma (check first)
    for i in range(n):
        if abs(values[i] - mean) > 3 * sigma:
            rules_violations[1].append(i)
            checked_points.add(i)

    # Rule 2: 9 points in a row on same side
    for i in range(n-8):
        end_point = i + 8
        if end_point not in checked_points:
            points = values[i:i+9]
            if all(x > mean for x in points) or all(x < mean for x in points):
                rules_violations[2].append(end_point)
                checked_points.add(end_point)

    # Rule 3: 6 points in a row, all increasing or decreasing
    for i in range(n-5):
        end_point = i + 5
        if end_point not in checked_points:
            points = values[i:i+6]
            if all(points[j] < points[j+1] for j in range(5)) or all(points[j] > points[j+1] for j in range(5)):
                rules_violations[3].append(end_point)
                checked_points.add(end_point)

    # Rule 4: 14 points alternating
    for i in range(n-13):
        end_point = i + 13
        if end_point not in checked_points:
            points = values[i:i+14]
            if all((points[j] - points[j+1]) * (points[j+1] - points[j+2]) < 0 for j in range(12)):
                rules_violations[4].append(end_point)
                checked_points.add(end_point)

    # Rule 5: 2 out of 2 or 2 out of 3 points > 2 sigma
    for i in range(n-1):  # Changed range to n-1 to allow checking pairs
        # Check 2/2 pattern
        if i >= 0:  # Changed condition since we always have enough points for [0,1] pattern
            end_point = i + 1
            if end_point not in checked_points:
                points_2 = values[i:i+2]
                if (all(x > mean + 2 * sigma for x in points_2)):
                    rules_violations[5].append(end_point)
                    checked_points.add(end_point)
                    continue
                elif (all(x < mean - 2 * sigma for x in points_2)):
                    rules_violations[5].append(end_point)
                    checked_points.add(end_point)
                    continue

        # Check 2/3 pattern only if 2/2 pattern wasn't found
        if i < n-2 and (i+2) not in checked_points:
            points_3 = values[i:i+3]
            above_2sigma = [x > mean + 2 * sigma for x in points_3]
            below_2sigma = [x < mean - 2 * sigma for x in points_3]

            if (sum(above_2sigma) >= 2 and points_3[-1] > mean + 2 * sigma):
                rules_violations[5].append(i+2)
                checked_points.add(i+2)
            elif (sum(below_2sigma) >= 2 and points_3[-1] < mean - 2 * sigma):
                rules_violations[5].append(i+2)
                checked_points.add(i+2)

    # Rule 6: 4 out of 4 or 4 out of 5 points > 1 sigma
    for i in range(n-4):
        # Check 4/4 pattern
        end_point = i + 3
        if end_point not in checked_points:
            points_4 = values[i:i+4]
            if (all(x > mean + sigma for x in points_4) and
                values[end_point] > mean + sigma):
                rules_violations[6].append(end_point)
                checked_points.add(end_point)
                continue  # Skip 4/5 check if 4/4 pattern found
            elif (all(x < mean - sigma for x in points_4) and
                  values[end_point] < mean - sigma):
                rules_violations[6].append(end_point)
                checked_points.add(end_point)
                continue  # Skip 4/5 check if 4/4 pattern found

        # Check 4/5 pattern
        end_point = i + 4
        if end_point not in checked_points:
            points_5 = values[i:i+5]
            above_1sigma = [x > mean + sigma for x in points_5]
            below_1sigma = [x < mean - sigma for x in points_5]

            if (sum(above_1sigma) >= 4 and points_5[-1] > mean + sigma):
                rules_violations[6].append(end_point)
                checked_points.add(end_point)
            elif (sum(below_1sigma) >= 4 and points_5[-1] < mean - sigma):
                rules_violations[6].append(end_point)
                checked_points.add(end_point)

    # Rule 7: 15 points within 1 sigma
    for i in range(n-14):
        end_point = i + 14
        if end_point not in checked_points:
            points = values[i:i+15]
            if all(abs(x - mean) <= sigma for x in points):
                rules_violations[7].append(end_point)
                checked_points.add(end_point)

    # Rule 8: 8 points > 1 sigma
    for i in range(n-7):
        end_point = i + 7
        if end_point not in checked_points:
            points = values[i:i+8]
            if all(abs(x - mean) > sigma for x in points):
                rules_violations[8].append(end_point)
                checked_points.add(end_point)

    return rules_violations, checked_points

def calculate_imr_limits(values):
    moving_ranges = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    mean = np.mean(values)
    mr_mean = np.mean(moving_ranges)
    d2 = 1.128  # Constant for n=2

    estimated_sd = mr_mean / d2
    ucl = mean + 3 * estimated_sd
    lcl = mean - 3 * estimated_sd

    return {
        'mean': mean,
        'estimated_sd': estimated_sd,
        'ucl': ucl,
        'lcl': lcl
    }

def plot_multiple_icharts(columns, data, xname='X', yname='Y', header='(I-Chart)', stage=' '):
    # Use the specified stage for grouping, or blank if stage is empty
    group_column = stage if stage.strip() else None

    # Calculate number of rows and columns for subplot grid
    n_plots = len(columns)
    n_rows = int(np.ceil(n_plots / 1))  # 1 column of plots
    n_cols = min(1, n_plots)  # Maximum 1 column

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 7 * n_rows))

    for idx, column in enumerate(columns, 1):
        # Create subplot
        ax = plt.subplot(n_rows, n_cols, idx)

        if group_column:
            # Group by the specified column (e.g., 'Fiscal year')
            unique_groups = data[group_column].unique()
        else:
            # Treat all data as a single group if no grouping column is specified
            unique_groups = ['All']
            data['Group'] = 'All'

        for group in unique_groups:
            stage_data = data if not group_column else data[data[group_column] == group]
            months = stage_data['Month']
            values = stage_data[column].values

            # Calculate IMR limits
            imr_stats = calculate_imr_limits(values)

            # Plot control lines
            ax.plot(months, values, '-', color='#0054A6', label='Values', linewidth=0.5)
            ax.plot(months, [imr_stats['mean']] * len(months), linestyle='-', color='#00841F', label='Mean', linewidth=0.5)
            ax.plot(months, [imr_stats['ucl']] * len(months), linestyle='-', color='#931313', label='UCL', linewidth=0.5)
            ax.plot(months, [imr_stats['lcl']] * len(months), linestyle='-', color='#931313', label='LCL', linewidth=0.5)

            # Add labels for UCL, LCL, and Mean only at the last point of the last group
            if group == unique_groups[-1]:
                last_month = months.iloc[-1]
                ax.text(
                    last_month, imr_stats['ucl'],
                    f'            UCL: {imr_stats["ucl"]:.4f}\n\n',
                    color='black', fontsize=11, va='center', ha='left')

                ax.text(
                    last_month, imr_stats['mean'],
                    f'            Mean: {imr_stats["mean"]:.4f}',
                    color='black', fontsize=11, va='center', ha='left')

                ax.text(
                    last_month, imr_stats['lcl'],
                    f'\n\n            LCL: {imr_stats["lcl"]:.4f}',
                    color='black', fontsize=11, va='center', ha='left')

            # Check Nelson rules
            rules_violations, checked_points = check_nelson_rules(values, imr_stats['mean'], imr_stats['estimated_sd'])

            # Plot all points as blue circles first
            ax.plot(months, values, 'o', color='#0054A6', markersize=8)

            # Plot violations (red squares) with rule numbers
            for rule, indices in rules_violations.items():
                for i in indices:
                    ax.plot(months.iloc[i], values[i], 's', color='#CE0000', markersize=8)

                    # Determine text position based on point position relative to mean
                    y_offset = 7 if values[i] > imr_stats['mean'] else -17

                    ax.annotate(
                        str(rule),
                        (months.iloc[i], values[i]),
                        xytext=(0, y_offset),
                        textcoords='offset points',
                        fontsize=11,
                        color='black',
                        ha='center'
                    )

            # Add group transitions
            if group != unique_groups[-1]:
                ax.axvline(x=months.iloc[-1] + pd.Timedelta(days=15),
                           color='#754DBD', linestyle=(0, (5, 5)), linewidth=0.5)

        top_y = ax.get_ylim()[1]

        # Add group label at the first point
        for group in unique_groups:
            stage_data = data if not group_column else data[data[group_column] == group]
            months = stage_data['Month']
            ax.annotate(
                f'{group}',
                xy=(months.iloc[0], top_y),  # Position at the top y-limit for the first month
                xytext=(0, 3),  # Offset slightly above the top
                textcoords='offset points',
                fontsize=11,
                color='black',
                ha='left',
                va='bottom'
            )

        # Set titles and labels
        ax.set_title(f'{column} {header}\n', fontsize=15)
        ax.set_xlabel(xname, fontsize=13)
        ax.set_ylabel(yname, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)

        # Set custom ticks and format
        custom_dates = pd.date_range(start=data['Month'].min(), end=data['Month'].max(), freq='6MS')
        ax.set_xticks(custom_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))

        # Rotate and style labels
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=11, ha='right')

    plt.tight_layout()
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Control Chart Plotter")
    
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.dataframe(data)
        columns = st.multiselect("Select columns to plot", options=data.columns)
        if columns:
            xname = st.text_input("X-axis Label", "X")
            yname = st.text_input("Y-axis Label", "Y")
            header = st.text_input("Chart Header", "(I-Chart)")
            # stage = st.text_input("Grouping Stage (optional)", "")
            options = [""] + list(data.columns)
            stage = st.selectbox("Grouping Stage (optional)", options=options, index=0)
            
            plot_multiple_icharts(columns, data, xname, yname, header, stage)

if __name__ == "__main__":
    main()
