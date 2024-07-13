#!/bin/bash

# Define the directory to search
search_dir="code"

# Define the output file
output_file="combined_script.py"

# Create or empty the output file
> "$output_file"

# Function to combine .py files
combine_py_files() {
  local dir="$1"
  for file in "$dir"/*.py; do
    if [ -f "$file" ] && [ "$file" != "$output_file" ] && [[ "$(basename "$file")" != "__init__.py" ]]; then
      echo "# File: $file" >> "$output_file"
      cat "$file" >> "$output_file"
      echo -e "\n" >> "$output_file"
    fi
  done

  for sub_dir in "$dir"/*; do
    if [ -d "$sub_dir" ]; then
      combine_py_files "$sub_dir"
    fi
  done
}

# Start combining from the specified directory
combine_py_files "$search_dir"

echo "All Python files have been combined into $output_file"
