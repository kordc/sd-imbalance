import json

input_file = 'metadata.jsonl'
output_file = 'output.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Load each JSON object (each line should be valid JSON)
        data = json.loads(line)
        # If the key exists, change its value to an empty string
        if "prompt" in data:
            data["prompt"] = ""
        # Write the modified object back as a JSON line
        outfile.write(json.dumps(data) + "\n")
