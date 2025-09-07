#!/usr/bin/env python3
import pandas as pd
import os

def merge_csv_files(ar_file, llada_file, output_file):
    """Merge AR models and LLaDA model results into one CSV file"""
    # Read AR models results
    ar_df = pd.read_csv(ar_file)
    
    # Read LLaDA results
    llada_df = pd.read_csv(llada_file)
    
    # Combine the dataframes
    combined_df = pd.concat([ar_df, llada_df], ignore_index=True)
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    print(f"Merged {ar_file} and {llada_file} -> {output_file}")

def main():
    # Define constraint types
    constraint_types = ['content', 'situation', 'style', 'format']
    
    # Define metric types
    metric_types = ['ssr', 'hsr', 'csl']
    
    # Create backup directory for current results
    os.makedirs('evaluation_result_ar_only', exist_ok=True)
    
    # Copy current AR results to backup
    for constraint in constraint_types:
        for metric in metric_types:
            filename = f"{constraint}_{metric}.csv"
            if os.path.exists(f"evaluation_result/{filename}"):
                os.system(f"cp evaluation_result/{filename} evaluation_result_ar_only/")
    
    # Merge with LLaDA results from backup
    for constraint in constraint_types:
        for metric in metric_types:
            ar_file = f"evaluation_result/{constraint}_{metric}.csv"
            llada_file = f"evaluation_result_backup/{constraint}_{metric}.csv"
            output_file = f"evaluation_result/{constraint}_{metric}.csv"
            
            if os.path.exists(ar_file) and os.path.exists(llada_file):
                merge_csv_files(ar_file, llada_file, output_file)
            elif os.path.exists(llada_file):
                # If only LLaDA file exists, copy it
                os.system(f"cp {llada_file} {output_file}")
                print(f"Copied {llada_file} -> {output_file}")

if __name__ == "__main__":
    main()
