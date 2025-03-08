import pandas as pd
import openai
import csv

# Configure OpenAI API
client = openai.OpenAI(
    api_key="KEY"
)

### 1️ Load Data from CSV with Custom Columns ###
def load_data(file_path: str) -> pd.DataFrame:
    """Load the spell-check dataset from CSV with custom columns."""
    
    # Define the column names for your custom CSV
    column_names = [
        'query', 
        'suggested_query_SCS', 'SCS_Correct', 
        'suggested_query_BERT', 'BERT_Correct', 
        'suggested_query_LanguageTool', 'LanguageTool_Correct', 
        'suggested_query_Bing', 'Bing_Correct', 
        'Query_Misspelled'
    ]
    
    # Read CSV using pandas (skip header if it exists)
    try:
        df = pd.read_csv(file_path, names=column_names, skiprows=1)
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        # Try alternative reading method
        df = pd.read_csv(file_path)
        # If columns don't match, rename them
        if len(df.columns) == len(column_names):
            df.columns = column_names
    
    # Create derived columns
    
    # 1. Best raw spelling (prioritize corrections that are marked as correct)
    df['llm_raw_spelling'] = df.apply(get_best_spelling_correction, axis=1)
    
    # 2. Original query misspelled flag (convert TRUE/FALSE to 1/0)
    df['llm_original_misspelled'] = df['Query_Misspelled'].apply(
        lambda x: 1 if str(x).upper() == 'TRUE' else 0
    )
    
    # 3. For each spellcheck system, create a column for correction accuracy
    for system in ['SCS', 'BERT', 'LanguageTool', 'Bing']:
        column_name = f'llm_correction_accurate_{system}'
        df[column_name] = df[f'{system}_Correct'].apply(
            lambda x: 1 if str(x).upper() == 'TRUE' else 0
        )
    
    # 4. Add overall correction accuracy
    df['llm_correction_accurate'] = df.apply(
        lambda row: 1 if any([
            row[f'llm_correction_accurate_{system}'] == 1 
            for system in ['SCS', 'BERT', 'LanguageTool', 'Bing']
        ]) else 0,
        axis=1
    )
    
    return df

def get_best_spelling_correction(row):
    """
    Select the best spelling correction based on which systems were correct.
    Prioritize BERT, then Bing, then LanguageTool, then SCS.
    """
    if str(row['BERT_Correct']).upper() == 'TRUE':
        return row['suggested_query_BERT']
    elif str(row['Bing_Correct']).upper() == 'TRUE':
        return row['suggested_query_Bing']
    elif str(row['LanguageTool_Correct']).upper() == 'TRUE':
        return row['suggested_query_LanguageTool']
    elif str(row['SCS_Correct']).upper() == 'TRUE':
        return row['suggested_query_SCS']
    else:
        # If none are correct, return the original query
        return row['query']

### 2️Use LLM to Analyze Spellcheck Accuracy ###
def analyze_spelling(row: pd.Series) -> dict:
    """Analyze spelling correction and determine if it was accurate."""
    prompt = f"""
    Given the following spelling correction:
    - Original: {row['query']}
    - Spellchecker Output: {row['llm_raw_spelling']}
    
    Answer the following questions:
    1. Was the original query misspelled? (Yes/No)
    2. Was the spellcheck correction accurate? (Yes/No)
    3. Classify the case:
       - True Positive (TP): Original was misspelled AND corrected correctly.
       - False Positive (FP): Original was correct BUT spellchecker made an unnecessary change.
    Respond in CSV format: is_misspelled,spellcheck_correct,label
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a spell-check analysis assistant. Respond only in CSV format (is_misspelled,spellcheck_correct,label)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip().split(',')
        if len(result) == 3:
            return {
                'is_misspelled': result[0].strip(),
                'spellcheck_correct': result[1].strip(),
                'label': result[2].strip()
            }
        else:
            raise ValueError(f"Invalid response format: {result}")
    
    except Exception as e:
        print(f"Error processing row: {row['query']} - {str(e)}")
        return {
            'is_misspelled': 'Error',
            'spellcheck_correct': 'Error',
            'label': 'Error'
        }

### 3 Calculate Precision ###
def calculate_precision(df: pd.DataFrame) -> float:
    """Calculate precision: TP / (TP + FP)."""
    true_positives = (df['label'] == 'True Positive').sum()
    false_positives = (df['label'] == 'False Positive').sum()
    
    if (true_positives + false_positives) == 0:
        return 0.0  # Avoid division by zero
    
    precision = true_positives / (true_positives + false_positives)
    return round(precision, 3)

### 4️ Main Execution ###
def main():
    input_file = '/Users/thomaslorenc/Sites/eyes/data-scripts-main/data-pull/src/myenv/cyber/cognitive_load_model/indeed/SpellcheckBIG.csv'  # Update with your actual file path
    df = load_data(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Optional: Run GPT analysis on the data
    run_gpt_analysis = True  # Set to False to skip GPT analysis
    
    if run_gpt_analysis:
        # Analyze spell-check accuracy using LLM
        print("Analyzing with GPT-4 (this may take a while)...")
        results = df.apply(analyze_spelling, axis=1).apply(pd.Series)
        
        # Merge LLM results with the original dataframe
        df = pd.concat([df, results], axis=1)
        
        # Add yes/no columns based on LLM labels if they don't exist already
        if 'llm_original_misspelled' not in df.columns:
            df['llm_original_misspelled'] = df['is_misspelled'].apply(
                lambda x: 1 if str(x).lower() == 'yes' else 0
            )
        
        if 'llm_correction_accurate' not in df.columns:
            df['llm_correction_accurate'] = df['spellcheck_correct'].apply(
                lambda x: 1 if str(x).lower() == 'yes' else 0
            )
        
        # Compute precision
        precision = calculate_precision(df)
        print(f"\nSpellcheck Precision: {precision}")
    
    # Reorder columns for better readability
    # Put the most important columns first, followed by others
    important_columns = [
        'query', 
        'llm_raw_spelling',
        'llm_original_misspelled',
        'llm_correction_accurate'
    ]
    
    if run_gpt_analysis:
        important_columns.extend(['is_misspelled', 'spellcheck_correct', 'label'])
    
    important_columns.extend([
        'llm_correction_accurate_SCS',
        'llm_correction_accurate_BERT',
        'llm_correction_accurate_LanguageTool',
        'llm_correction_accurate_Bing'
    ])
    
    other_columns = [col for col in df.columns if col not in important_columns]
    column_order = important_columns + other_columns
    
    df = df[column_order]
    
    # Save the analyzed results
    output_file = "spellcheck_analysis_results.csv"
    df.to_csv(output_file, index=False)
    
    # Print summary (limited to first 10 rows for readability)
    print("\nAnalysis Results (first 10 rows):")
    print(df[important_columns].head(10))
    print(f"\nResults saved to '{output_file}'")

if __name__ == "__main__":
    main()