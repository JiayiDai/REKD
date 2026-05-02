import os
import json
import glob

def calculate_mednli_word_counts():
    # Find all jsonl files in the specified directory
    #file_pattern = os.path.join(data_folder, '*.jsonl')
    file_pattern = '*.jsonl'
    jsonl_files = glob.glob(file_pattern)
    
    if not jsonl_files:
        print(f"No .jsonl files found. If your data is in CSV format, please adjust the script to use pandas.")
        return

    word_counts = []

    for file_path in ["mli_train_v1.jsonl"]:
        print(f"Processing {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Extract the two sentences (defaulting to empty string if missing)
                    sentence1 = data.get('sentence1', '')
                    sentence2 = data.get('sentence2', '')
                    # Split sentences by whitespace to get words
                    combined_words = sentence1.split() + sentence2.split()
                    
                    # Append the total number of words for this sample
                    word_counts.append(len(combined_words))
                    
                except json.JSONDecodeError:
                    print("Skipping invalid JSON line.")
                    continue

    # Print summary statistics
    if word_counts:
        total_samples = len(word_counts)
        avg_words = sum(word_counts) / total_samples
        max_words = max(word_counts)
        min_words = min(word_counts)
        
        print("\n--- Word Count Summary ---")
        print(f"Total samples processed: {total_samples}")
        print(f"Average words per sample: {avg_words:.2f}")
        print(f"Minimum words in a sample: {min_words}")
        print(f"Maximum words in a sample: {max_words}")
    else:
        print("No samples were successfully processed.")

# Run the function on your specific folder
#data_dir = 'data/med_nli'
calculate_mednli_word_counts()