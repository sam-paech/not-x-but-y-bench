#!/usr/bin/env python3
import json
import glob
from pathlib import Path

def filter_samples(file_path):
    """Remove samples with prompt_index >= 150 from a results file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Process each model's results in the file
    modified = False
    for model_key in data:
        if 'samples' in data[model_key]:
            original_count = len(data[model_key]['samples'])
            # Keep only samples with prompt_index < 150
            # Handle cases where prompt_index might be None
            data[model_key]['samples'] = [
                sample for sample in data[model_key]['samples']
                if sample.get('prompt_index') is not None and sample.get('prompt_index') < 150
            ]
            new_count = len(data[model_key]['samples'])
            if original_count != new_count:
                modified = True
                print(f"{file_path.name}: Removed {original_count - new_count} samples (kept {new_count})")

    # Write back to file if modified
    if modified:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    return False

def main():
    results_dir = Path('results')
    json_files = list(results_dir.glob('*.json'))

    print(f"Processing {len(json_files)} result files...\n")

    total_modified = 0
    for json_file in sorted(json_files):
        if filter_samples(json_file):
            total_modified += 1

    print(f"\n✓ Processed {len(json_files)} files, modified {total_modified} files")

if __name__ == '__main__':
    main()
