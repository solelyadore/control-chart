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
2019-10-01,32.75465466,85.419001374,43.351748832,5.137985062,0.321124048,1.926744386,39648,2020
2019-11-01,43.856545166,44.536491608,30.937562866,4.079678603,0.679946442,3.739705382,37533,2020
2019-12-01,44.498871711,15.844295236,23.597886529,4.045351996,1.038768836,2.359788648,37838,2020
2020-01-01,34.033869191,12.631126704,32.279546005,2.105187784,1.403458539,2.456052431,36414,2020
2020-02-01,30.938952163,14.948033051,21.205349221,4.171544097,0.695257374,0.347628687,36739,2020
2020-03-01,33.476270065,17.094265587,30.983356355,2.84904424,0.71226106,1.78065265,35898,2020
2020-04-01,46.65544604,14.609281057,26.390959343,2.827602771,0.942534257,2.356335667,27494,2020
2020-05-01,37.649752259,23.819230995,26.508499024,3.457630316,0.5969355175,1.920905742,33386,2020
2020-06-01,36.493579962,20.001289028,30.879183041,2.807198436,1.403599218,3.508998094,36411,2020
2020-07-01,34.543873294,16.141996843,31.638313875,6.133958817,0.64567986,4.519759118,39445,2020
2020-08-01,35.442338813,35.120135736,25.131840251,4.18864005,0.644406154,1.933218462,39520,2020
2020-09-01,32.007943219,15.683892168,20.165004202,4.481112034,1.600397183,3.8409532,39772,2020
2020-10-01,31.504897802,25.945209969,16.679063548,3.706458588,0.617743098,2.162100843,41161,2021
2020-11-01,32.233841206,19.461941863,20.982406081,3.953206957,0.304092824,2.432742739,41784,2021
2020-12-01,29.6461221,30.535505763,19.269979365,3.853995873,0.592922442,1.185844884,42821,2021
2021-01-01,34.171555173,11.055503137,8.0403659,1.675076221,0.670030508,3.015137237,38066,2021
2021-02-01,34.274222913,18.565204084,14.994972552,1.785115766,0.888286275333331,2.856185255,35812,2021
2021-03-01,31.917665472,19.150599293,24.622199042,2.735799899,0.911933316,2.735799899,41799,2021
2021-04-01,32.453211511,24.423550935,15.05561358,2.007415144,1.12676651826666,3.011122716,38115,2021
2021-05-01,32.94804296,16.80350189,6.260128182,2.306363017,0.32948041,2.965323886,38680,2021
2021-06-01,31.307250471,10.222775675,9.58385218,1.916770436,0.206940508755554,4.79192609,39846,2021
2021-07-01,37.561770211,17.774766268,8.719696657,2.682983583,0.109138073924443,3.018356537,38027,2021
2021-08-01,39.74926256,15.538348105,7.227138625,2.89085545,0.361356919,1.084070806,35400,2021
2021-09-01,36.830500973,17.889100467,10.873766939,2.455366725,0.911933316,1.753833382,36424,2021
2021-10-01,31.123908837,12.745981724,8.596127183,4.149854492,1.12676651826666,0.889254548,42827,2022
2021-11-01,31.341102954,12.866347529,14.185972899,6.598126948,1.34159972053332,1.31962537,38632,2022
2021-12-01,31.201358433,14.182435638,15.443096613,2.206156645,1.55643292279999,0.945495719,40369,2022
2022-01-01,29.724086392,19.398877442,9.073668492,3.441736333,0.938655368,1.564425597,40652,2022
2022-02-01,29.390288445,15.014603863,8.305951094,5.430814158,0.63891933,0.958378946,39846,2022
2022-03-01,31.08702737,22.816166891,11.69328552,3.99282919,0.855606248,2.852020843,44452,2022
2022-04-01,31.682772555,12.961134256,7.776680534,4.320378069,1.072293166,2.016176442,44031,2022
2022-05-01,28.982919399,22.967596491,8.202713043,6.562170405,0.546847546,3.007661454,46303,2022
2022-06-01,32.250292368,26.203362549,33.402088524,8.63847117,0.863847117,2.87949039,44043,2022
2022-07-01,30.305373,31.387707785,64.669501358,4.599922677,0.541167368,0.270583684,46773,2022
2022-08-01,31.01026887,24.538560599,49.346775695,5.662744731,0.269654497,1.348272583,46929,2022
2022-09-01,31.856628769,12.549581002,28.63878745,4.826761954,0.321784127,1.608920635,39569,2022
2022-10-01,27.3549115,0.8824165,18.5307465,2.941388317,0.158152089666667,2.353110683,43147,2023
2022-11-01,30.818169172,20.446669922,38.226382936,3.852271171,0.296328529,3.852271171,42840,2023
2022-12-01,29.430429931,7.774075841,29.708075544,0.555291128,0.832936692,2.498810076,45621,2023
2023-01-01,28.572389755,9.340973586,29.396593275,0.824203569,0.549469046,2.197876135,46089,2023
2023-02-01,27.847831508,10.442936791,32.199055175,3.480978914,0.290081568,1.160326321,43730,2023
2023-03-01,26.067364176,8.689121392,12.266994915,1.277811955,0.0146406446666665,0.766687173,49434,2023
2023-04-01,24.847192783,8.816745812,20.572406911,2.67174117,0.267174117,1.335870585,47351,2023
2023-05-01,24.571667351,6.14291685,17.148976194,2.04763895,0.255954881,1.023819475,49360,2023
2023-06-01,25.400418953,5.183758951,14.514525102,3.887819201,0.5183759,1.29593975,48763,2023
2023-07-01,24.976204491,6.11662149,12.742961462,1.274296156,1.019436915,1.274296156,49566,2023
2023-08-01,26.055621816,9.865720858,14.41913053,1.770770428,1.011868816,1.517803224,49926,2023
2023-09-01,26.690114535,4.575448206,22.368857896,1.270957835,1.343386793,1.016766268,49692,2023
2023-10-01,22.887866337,8.035102026,16.557179919,2.9218553,0.243487958,1.704415559,51811,2024
2023-11-01,24.699486154,11.845671943,12.349743077,4.03256917,0.504071134,1.512213451,50105,2024
2023-12-01,23.902074658,3.485719223,10.706137596,4.979598883,0.76465431,0.995919757,50701,2024
2024-01-01,23.968938196,7.989646049,17.976703647,3.245793712,1.025237486,1.9974115,50564,2024
2024-02-01,25.536963725,7.335085331,20.918576672,2.716698278,0.271669818,2.988368096,46592,2024
2024-03-01,22.434804644,1.909345074,28.401507988,1.909345074,0.477336293,1.43200883,52827,2024
2024-04-01,138.613861386,0.742574273,20.297029704,2.970297043,0.683002768,0.742574273,50990,2024
2024-05-01,156.748649715,1.162823802,20.465699072,1.860518093,0.23256478,2.558212384,54174,2024
2024-06-01,155.44127165,1.907254881,18.834141809,2.860882297,0.476813708,2.145661735,52883,2024
2024-07-01,183.41722238,16.24934129,27.156433423,1.558155998,0.721062636,1.335562277,56533,2024
2024-08-01,178.108965454,1.320956504,50.636665072,3.302391211,0.965311564,2.86207236,57142,2024
2024-09-01,168.1275848,17.03692858,52.455806487,3.810891945,0.448340249,1.345020698,56146,2024
2024-10-01,166.989656967,5.624004008,44.992031917,1.946770588,0.216307854,1.730462783,58133,2025
2024-11-01,173.565383173,11.451273638,45.131490327,2.694417341,0.224534758,1.122673888,56058,2025
"""

# Function to load example data as DataFrame
def load_example_data():
    return pd.read_csv(StringIO(EXAMPLE_DATA), parse_dates=['Month'])

# Streamlit App
def main():
    st.title("Control Chart Plotter")

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

        # Format 'Month' and 'Fiscal year'
        data['Month'] = data['Month'].dt.strftime('%b %Y')  # Format Month as "Oct 2019"
        data['Fiscal year'] = data['Fiscal year'].astype(int)  # Display Fiscal year without commas

        # Display the formatted data
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
