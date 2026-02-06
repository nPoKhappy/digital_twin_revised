# 這個腳本用於從之前分析的敏感性結果中，根據每個輸入變量對未來預測的平均影響，繪製一個條形圖來展示各個變量的重要性。圖中將按照工藝流程的順序排列變量，並使用顏色區分影響力較大的前10個變量（紅色）和影響力較小的後10個變量（綠色），其他變量則使用淺藍色。這樣可以清晰地展示哪些變量對預測結果有較大影響，以及它們在工藝流程中的位置。
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# 設定字體以顯示中文 (如果有的話)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_all_ranking(target_name, result_dir):
    csv_path = os.path.join(result_dir, f'heatmap_{target_name}.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Load data (Index is Time Lag tau, Columns are Variables)
    df = pd.read_csv(csv_path, index_col=0)
    
    # Calculate total impact score for each variable (Sum over time lags)
    # The new CSV has columns "t+1"..."t+18" (Future time steps), index is Variable
    # We want to use the mean across future time steps as the score
    total_impact = df.mean(axis=1)
    
    # Process Order (Claus Process Flow: Input -> Output)
    process_order = [
            'acidgas_Fm', 'acidgas_CO2', 'acidgas_H2O', 'acidgas_H2S', 'acidgas_T', 'acidgas_P', 
            'air', 'air_SP', 'second_air2', 'air2_SP', 'COG', 'COG_SP',
            'burner_input_T_SP', 'burner_input_T_PV', 'burner_inputP', 
            'burner_output_T_SP', 'burner_output_T_PV', 'burner_output_P_SP', 'burner_output_P_PV', 
            'fur_F', 'fur_inputT', 'fur_inputP', 'fur_temp', 'fur_outputT', 'fur_outputP_SP', 'fur_outputP_PV', 
            'WHB_F', 'WHB_inputT', 'WHB_inputP', 'WHB_outputT', 'WHB_outputP', 
            'SEP1_F', 'SEP1_P_SP', 'SEP1_P_PV', 'SEP1_T', 
            'HEATER1_F', 'HEATER1_input_T', 'HEATER1_input_P', 'HEATER1_output_T_SP', 'HEATER1_output_T_PV', 'HEATER1_output_P', 
            'cat1_F', 'cat1_input_temp', 'cat1_output_temp', 'cat1_input_P', 'cat1_output_P_SP', 'cat1_output_P_PV', 'cat1_deltaP', 
            'SEP2_F', 'SEP2_P_SP', 'SEP2_P_PV', 'SEP2_T', 
            'HEATER2_F', 'HEATER2_input_T', 'HEATER2_input_P', 'HEATER2_output_T_SP', 'HEATER2_output_T_PV', 'HEATER2_output_P', 
            'cat2_F', 'cat2_input_temp', 'cat2_output_temp', 'cat2_input_P', 'cat2_output_P_SP', 'cat2_output_P_PV', 'cat2_deltaP', 
            'SEP3_F', 'SEP3_P_SP', 'SEP3_P_PV', 'SEP3_T', 
            'B35_H2S', 'B35_SO2'
    ]
    
    # Reindex total_impact to match process_order
    # Filter only variables that exist in the result/impact data
    ordered_metrics = [v for v in process_order if v in total_impact.index]
    # Append any extra variables that might be in impact data but not in our list
    extra_metrics = [v for v in total_impact.index if v not in ordered_metrics]
    final_order = ordered_metrics + extra_metrics
    
    total_impact = total_impact.reindex(final_order)
    
    # Identify Top 10 Variables by magnitude (for highlighting)
    top_10_vars = total_impact.nlargest(10).index
    
    # Identify Bottom 10 Variables by magnitude (for highlighting)
    bottom_10_vars = total_impact.nsmallest(10).index
    
    # Define colors: Red for Top 10, Green for Bottom 10, Skyblue for others
    bar_colors = []
    for var in total_impact.index:
        if var in top_10_vars:
            bar_colors.append('red')
        elif var in bottom_10_vars:
            bar_colors.append('green')
        else:
            bar_colors.append('skyblue')
    
    # Create the plot
    # Use vertical bars to show "Left to Right" flow
    plt.figure(figsize=(20, 8)) 
    
    # Vertical bar plot
    bars = plt.bar(total_impact.index, total_impact.values, color=bar_colors)
    
    # Create a custom legend for the colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Top 10 High Impact Variables'),
                       Patch(facecolor='green', label='Bottom 10 Low Impact Variables'),
                       Patch(facecolor='skyblue', label='Other Variables')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'Sensitivity Analysis - Process Flow Order for {target_name}\n(Left -> Right: Upstream -> Downstream)', fontsize=16)
    plt.ylabel('Mean Impact Score (Avg Deviation per Unit Perturbation)', fontsize=14)
    plt.xlabel('Input Variable (Process Order)', fontsize=14)
    plt.xticks(rotation=90, fontsize=10) # Rotate x labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, f'process_order_ranking_{target_name}.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

def main():
    # Pointing to the explicit Training Set analysis folder
    result_dir = 'results/transformer_71var_vanilla/sensitivity_analysis_train_set'
    plot_all_ranking('B35_H2S', result_dir)
    plot_all_ranking('B35_SO2', result_dir)

if __name__ == "__main__":
    main()
