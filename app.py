import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from io import StringIO

def check_nelson_rules(values, mean, sigma, r1, r2, r3, r4, r5, r6, r7, r8):
    n = len(values)
    rules_violations = {i: [] for i in range(1, 9)}
    checked_points = set()  # Keep track of points that have already violated a rule

    # Rule 1: Points > r1 sigma
    for i in range(n):
        if abs(values[i] - mean) > r1 * sigma:
            rules_violations[1].append(i)
            checked_points.add(i)

    # Rule 2: r2 points in a row on same side
    for i in range(n - r2 + 1):
        end_point = i + r2 - 1
        if end_point not in checked_points:
            points = values[i:i + r2]
            if all(x > mean for x in points) or all(x < mean for x in points):
                rules_violations[2].append(end_point)
                checked_points.add(end_point)

    # Rule 3: r3 points in a row, all increasing or decreasing
    for i in range(n - r3 + 1):
        end_point = i + r3 - 1
        if end_point not in checked_points:
            points = values[i:i + r3]
            if all(points[j] < points[j + 1] for j in range(r3 - 1)) or all(points[j] > points[j + 1] for j in range(r3 - 1)):
                rules_violations[3].append(end_point)
                checked_points.add(end_point)

    # Rule 4: r4 points alternating
    for i in range(n - r4 + 1):
        end_point = i + r4 - 1
        if end_point not in checked_points:
            points = values[i:i + r4]
            if all((points[j] - points[j + 1]) * (points[j + 1] - points[j + 2]) < 0 for j in range(r4 - 2)):
                rules_violations[4].append(end_point)
                checked_points.add(end_point)

    # Rule 5: r5 out of r5+1 points > 2 sigma
    for i in range(n - (r5 + 1) + 1):
        end_point = i + (r5 + 1) - 1
        if end_point not in checked_points:
            points = values[i:i + (r5 + 1)]
            above_2sigma = [x > mean + 2 * sigma for x in points]
            below_2sigma = [x < mean - 2 * sigma for x in points]

            if sum(above_2sigma) >= r5 or sum(below_2sigma) >= r5:
                rules_violations[5].append(end_point)
                checked_points.add(end_point)

    # Rule 6: r6 out of r6+1 points > 1 sigma
    for i in range(n - (r6 + 1) + 1):
        end_point = i + (r6 + 1) - 1
        if end_point not in checked_points:
            points = values[i:i + (r6 + 1)]
            above_1sigma = [x > mean + sigma for x in points]
            below_1sigma = [x < mean - sigma for x in points]

            if sum(above_1sigma) >= r6 or sum(below_1sigma) >= r6:
                rules_violations[6].append(end_point)
                checked_points.add(end_point)

    # Rule 7: r7 points within 1 sigma
    for i in range(n - r7 + 1):
        end_point = i + r7 - 1
        if end_point not in checked_points:
            points = values[i:i + r7]
            if all(abs(x - mean) <= sigma for x in points):
                rules_violations[7].append(end_point)
                checked_points.add(end_point)

    # Rule 8: r8 points > 1 sigma
    for i in range(n - r8 + 1):
        end_point = i + r8 - 1
        if end_point not in checked_points:
            points = values[i:i + r8]
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

def plot_multiple_icharts(columns, data, xname='X', yname='Y', header='(I-Chart)', stage='', r1=3, r2=9, r3=6, r4=14, r5=2, r6=4, r7=15, r8=8):
    # Create a copy of the data to avoid modifying the original
    plot_data = data.copy()
    
    # Use the specified stage for grouping, or blank if stage is empty
    group_column = stage if stage.strip() else None

    # Calculate number of rows and columns for subplot grid
    n_plots = len(columns)
    n_rows = int(np.ceil(n_plots / 1))  # 1 column of plots
    n_cols = min(1, n_plots)  # Maximum 1 column

    # Create figure - adjust height based on number of plots
    # Increase base height per plot and add extra space for test results
    fig = plt.figure(figsize=(12, 11 * n_rows))

    # Create GridSpec to control subplot layout
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[1] * n_rows, hspace=1.0)

    for idx, column in enumerate(columns):
        # Create subplot using GridSpec
        ax = fig.add_subplot(gs[idx])

        if group_column:
            unique_groups = plot_data[group_column].unique()
        else:
            unique_groups = ['All']

        # Initialize combined rules violations across all groups
        combined_violations = {i: [] for i in range(1, 9)}
        point_counter = 0

        for group in unique_groups:
            if group_column:
                stage_data = plot_data[plot_data[group_column] == group]
            else:
                stage_data = plot_data
                
            months = stage_data['Month']
            values = stage_data[column].values

            # Calculate IMR limits
            imr_stats = calculate_imr_limits(values)

            # Plot control lines
            ax.plot(months, values, '-', color='#0054A6', label='Values', linewidth=0.5)
            ax.plot(months, [imr_stats['mean']] * len(months), linestyle='-', color='#00841F', label='Mean', linewidth=0.5)
            ax.plot(months, [imr_stats['ucl']] * len(months), linestyle='-', color='#931313', label='UCL', linewidth=0.5)
            ax.plot(months, [imr_stats['lcl']] * len(months), linestyle='-', color='#931313', label='LCL', linewidth=0.5)

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

            rules_violations, checked_points = check_nelson_rules(values, imr_stats['mean'], imr_stats['estimated_sd'], r1, r2, r3, r4, r5, r6, r7, r8)

            ax.plot(months, values, 'o', color='#0054A6', markersize=8)

            for rule, indices in rules_violations.items():
                for i in indices:
                    ax.plot(months.iloc[i], values[i], 's', color='#CE0000', markersize=8)
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
                    combined_violations[rule].append(point_counter + i + 1)

            point_counter += len(values)

            if group != unique_groups[-1]:
                ax.axvline(x=months.iloc[-1] + pd.Timedelta(days=15),
                           color='#754DBD', linestyle=(0, (5, 5)), linewidth=0.5)

        # Generate test results summary
        if any(combined_violations.values()):
            test_results = []
            group_label = f" by {group_column}" if group_column else ""
            test_results.append(f"Test Results for I Chart of {column}{group_label}")
            test_results.append("")

            test_descriptions = {
                1: f"One point more than {r1} standard deviations from center line.",
                2: f"{r2} points in a row on the same side of center line.",
                3: f"{r3} points in a row all increasing or all decreasing.",
                4: f"{r4} points in a row alternating up and down.",
                5: f"{r5} out of {r5 + 1} points more than 2 standard deviations from center line (on one side of CL).",
                6: f"{r6} out of {r6 + 1} points more than 1 standard deviation from center line (on one side of CL).",
                7: f"{r7} points in a row within 1 standard deviation of center line (above and below CL).",
                8: f"{r8} points in a row more than 1 standard deviation from center line (above and below CL)."
            }
            
            for rule in sorted(combined_violations.keys()):
                if combined_violations[rule]:
                    points = [str(i) for i in combined_violations[rule]]
                    if points:
                        test_results.append(f"TEST {rule}. {test_descriptions[rule]}")
                        test_results.append(f"Test Failed at points:  {', '.join(points)}")
                        test_results.append("")
            
            # Add test results below each chart
            results_text = '\n'.join(test_results)
            plt.figtext(0.1, 1 - ((idx + 0.95) / n_rows), results_text,
                       fontsize=12, va='top', ha='left', wrap=True)

        top_y = ax.get_ylim()[1]

        for group in unique_groups:
            if group_column:
                stage_data = plot_data[plot_data[group_column] == group]
            else:
                stage_data = plot_data
                
            months = stage_data['Month']
            ax.annotate(
                f'{group}',
                xy=(months.iloc[0], top_y),
                xytext=(0, 3),
                textcoords='offset points',
                fontsize=11,
                color='black',
                ha='left',
                va='bottom'
            )

        ax.set_title(f'{column} {header}\n', fontsize=15)
        ax.set_xlabel(xname, fontsize=13)
        ax.set_ylabel(yname, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)

        custom_dates = pd.date_range(start=plot_data['Month'].min(), end=plot_data['Month'].max(), freq='6MS')
        ax.set_xticks(custom_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
        
        # Rotate and align x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust subplot spacing
        ax.set_position([0.1, 1 - ((idx + 0.85) / n_rows), 0.85, 0.7 / n_rows])

    # Don't use tight_layout() as we're manually positioning elements
    st.pyplot(fig)

# Example data
EXAMPLE_DATA = """
Month,Data1,Data2,Data3,Data4,Data5,Data6,N,Fiscal year
2019-10-01,0.66846234,1.743244926,0.884729568,0.104856838,0.006553552,0.039321314,152589,2020
2019-11-01,0.895031534,0.908907992,0.631378834,0.083258747,0.013876458,0.076320518,144129,2020
2019-12-01,0.908140239,0.323352964,0.481589521,0.082558204,0,0.048158952,145352,2020
2020-01-01,0.694568759,0.257778096,0.658766245,0.042963016,0.028642011,0.050123519,139655,2020
2020-02-01,0.631407187,0.305061899,0.432762229,0.085133553,0.014188926,0.007094463,140955,2020
2020-03-01,0.683189185,0.348862563,0.632313395,0.05814376,0.01453594,0.03633985,137590,2020
2020-04-01,0.95215196,0.298148593,0.538591007,0.057706179,0.019235393,0.048088483,103975,2020
2020-05-01,0.768362291,0.486106755,0.540989776,0.070563884,0,0.039202158,127544,2020
2020-06-01,0.744766938,0.408189572,0.630187409,0.057289764,0.028644882,0.071612206,139641,2020
2020-07-01,0.704977006,0.329428507,0.645679875,0.125182833,0.01317714,0.092239982,151778,2020
2020-08-01,0.723313037,0.716737464,0.512894699,0.08548245,0.013151146,0.039453438,152078,2020
2020-09-01,0.653223331,0.320079432,0.411530698,0.091451266,0.032661167,0.0783868,153087,2020
2020-10-01,0.642957098,0.529494081,0.340389052,0.075642012,0.012607002,0.044124507,158642,2021
2020-11-01,0.657833494,0.397182487,0.428212369,0.080677693,0.006205976,0.049647811,161135,2021
2020-12-01,0.6050229,0.623173587,0.393264885,0.078652977,0.012100458,0.024200916,165283,2021
2021-01-01,0.697378677,0.225622513,0.1640891,0.034185229,0.013674092,0.061533413,146262,2021
2021-02-01,0.699473937,0.378881716,0.306019848,0.036430934,0,0.058289495,137246,2021
2021-03-01,0.651380928,0.390828557,0.502493858,0.055832651,0.018610884,0.055832651,161196,2021
2021-04-01,0.662310439,0.498439815,0.30725742,0.040967656,0,0.061451484,146457,2021
2021-05-01,0.67240904,0.34292861,0.127757718,0.047068633,0.00672409,0.060516814,148719,2021
2021-06-01,0.638923479,0.208628075,0.19558882,0.039117764,0,0.09779441,153383,2021
2021-07-01,0.766566739,0.362750332,0.177952993,0.054754767,0,0.061599113,146106,2021
2021-08-01,0.81120944,0.317109145,0.147492625,0.05899705,0.007374631,0.022123894,135600,2021
2021-09-01,0.751642877,0.365083683,0.221913611,0.050109525,0,0.035792518,139694,2021
2021-10-01,0.635181813,0.260122076,0.175431167,0.084690908,0,0.018148052,165307,2022
2021-11-01,0.639614346,0.262578521,0.289509651,0.134655652,0,0.02693113,148527,2022
2021-12-01,0.636762417,0.289437462,0.315165237,0.045023605,0,0.019295831,155474,2022
2022-01-01,0.606614008,0.395895458,0.185176908,0.070239517,0.019156232,0.031927053,156607,2022
2022-02-01,0.599801805,0.306420487,0.169509206,0.110832942,0.01303917,0.019558754,153384,2022
2022-03-01,0.63442913,0.465636059,0.23863848,0.08148631,0.017461352,0.058204507,171808,2022
2022-04-01,0.646587195,0.264512944,0.158707766,0.088170981,0,0.041146458,170124,2022
2022-05-01,0.591488151,0.468726459,0.167402307,0.133921845,0.011160154,0.061380846,179209,2022
2022-06-01,0.658169232,0.534762501,0.681675276,0.17629533,0.017629533,0.05876511,170169,2022
2022-07-01,0.618477,0.640565465,1.319785742,0.093875973,0.011044232,0.005522116,181090,2022
2022-08-01,0.63286263,0.500786951,1.007077055,0.115566219,0.005503153,0.027515767,181714,2022
2022-09-01,0.650135281,0.256113898,0.58446505,0.098505346,0.006567023,0.032835115,152276,2022
2022-10-01,0.5582635,0.0180085,0.3781785,0.060028333,0,0.048022667,166588,2023
2022-11-01,0.628942228,0.417278978,0.780130264,0.078617779,0.006047521,0.078617779,165357,2023
2022-12-01,0.600621019,0.158654609,0.606287256,0.011332472,0.016998708,0.050996124,176484,2023
2023-01-01,0.583109995,0.190632114,0.599930475,0.016820481,0.011213654,0.044854615,178354,2023
2023-02-01,0.568323092,0.213121159,0.657123575,0.071040386,0.005920032,0.023680129,168918,2023
2023-03-01,0.531987024,0.177329008,0.250346835,0.026077795,0,0.015646677,191734,2023
2023-04-01,0.507085567,0.179933588,0.419845039,0.05452533,0.005452533,0.027262665,183401,2023
2023-05-01,0.501462599,0.12536565,0.349979106,0.04178855,0.005223569,0.020894275,191440,2023
2023-06-01,0.518375897,0.105790999,0.296214798,0.079343249,0.0105791,0.02644775,189052,2023
2023-07-01,0.509718459,0.12482901,0.260060438,0.026006044,0.020804835,0.026006044,192263,2023
2023-08-01,0.531747384,0.201341242,0.29426797,0.036138172,0.020650384,0.030975576,193701,2023
2023-09-01,0.544696215,0.093376494,0.456507304,0.025937915,0,0.020750332,192768,2023
2023-10-01,0.467099313,0.163981674,0.337901631,0.0596297,0.004969142,0.034783991,201242,2024
2023-11-01,0.504071146,0.241748407,0.252035573,0.08229733,0.010287166,0.030861499,194417,2024
2023-12-01,0.487797442,0.071137127,0.218492604,0.101624467,0,0.020324893,196803,2024
2024-01-01,0.489162004,0.163054001,0.366871503,0.066240688,0,0.0407635,196254,2024
2024-02-01,0.521162525,0.149695619,0.426909728,0.055442822,0.005544282,0.060987104,180366,2024
2024-03-01,0.457853156,0.038966226,0.579622612,0.038966226,0.009741557,0.02922467,205306,2024
2024-04-01,2.828854314,0.015154577,0.414225096,0.060618307,0,0.015154577,197960,2024
2024-05-01,3.198952035,0.023731098,0.417667328,0.037969757,0.00474622,0.052208416,210694,2024
2024-06-01,3.17227085,0.038923569,0.384370241,0.058385353,0.009730892,0.043789015,205531,2024
2024-07-01,3.74320862,0.33161921,0.554212927,0.031799102,0,0.027256373,220132,2024
2024-08-01,3.634876846,0.026958296,1.033401328,0.067395739,0,0.05840964,222566,2024
2024-09-01,3.4311752,0.34769242,1.070526663,0.077773305,0.009149801,0.027449402,218584,2024
2024-10-01,3.407952183,0.114775592,0.918204733,0.039730012,0.004414446,0.035315567,226529,2025
2024-11-01,3.542150677,0.233699462,0.921050823,0.054988109,0.004582342,0.022911712,218229,2025
"""

# Function to load example data as DataFrame
def load_example_data():
    return pd.read_csv(StringIO(EXAMPLE_DATA), parse_dates=['Month'])

# Streamlit App
def main():
    st.title("Control Chart Plotter")

    # Hidden Footer & Header
    # hide_st_style = """
    #         <style>
    #         #MainMenu {visibility: hidden;}
    #         footer {visibility: hidden;}
    #         header {visibility: hidden;}
    #         </style>
    #         """
    # st.markdown(hide_st_style, unsafe_allow_html=True)

    # Data storage for reactivity
    if "data" not in st.session_state:
        st.session_state.data = None

    # Button to load example file
    if st.button("Load Example File"):
        st.session_state.data = load_example_data()
    
    # File uploader for custom data
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        st.session_state.data = pd.read_excel(uploaded_file)

    # Display data if available
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Ensure 'Month' column exists and is in datetime format
        if 'Month' not in data.columns:
            st.error("The data must contain a 'Month' column.")
            return
        
        if not pd.api.types.is_datetime64_any_dtype(data['Month']):
            data['Month'] = pd.to_datetime(data['Month'], errors='coerce')
        
        st.dataframe(data)
        
        # Remove 'Month' from selectable columns for plotting
        plot_columns = [col for col in data.columns if col != 'Month']
        columns = st.multiselect("Select columns to plot", options=plot_columns)
        
        # Rule inputs
        st.sidebar.title("Rule Settings")
        r1 = st.sidebar.number_input("Rule 1: Points beyond Nσ", min_value=1, value=3, step=1)
        r2 = st.sidebar.number_input("Rule 2: Points on same side", min_value=1, value=9, step=1)
        r3 = st.sidebar.number_input("Rule 3: Points increasing or decreasing", min_value=2, value=6, step=1)
        r4 = st.sidebar.number_input("Rule 4: Points alternating", min_value=2, value=14, step=1)
        r5 = st.sidebar.number_input("Rule 5: N out of N+1 beyond 2σ", min_value=2, value=2, step=1)
        r6 = st.sidebar.number_input("Rule 6: N out of N+1 beyond 1σ", min_value=2, value=4, step=1)
        r7 = st.sidebar.number_input("Rule 7: Points within 1σ", min_value=2, value=15, step=1)
        r8 = st.sidebar.number_input("Rule 8: Points beyond 1σ", min_value=2, value=8, step=1)
        
        if columns:
            xname = st.text_input("X-axis Label", "X")
            yname = st.text_input("Y-axis Label", "Y")
            header = st.text_input("Chart Header", "(I-Chart)")
            # Include an optional grouping stage
            options = [""] + list(data.columns)
            stage = st.selectbox("Grouping Stage (optional)", options=options, index=0)
            
            plot_multiple_icharts(columns, data, xname, yname, header, stage, r1, r2, r3, r4, r5, r6, r7, r8)
    else:
        st.info("Please upload a file or load the example data.")

if __name__ == "__main__":
    main()