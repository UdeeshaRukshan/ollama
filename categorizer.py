import ollama
import os

model = "llama3.2"
input_file="./data/grocery_list.txt"
output_file="./data/categorized_grocery_list.txt"

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Input file'{input_file}' does not exist")
    exit(1)

#Read the uncategorized grocery items from the input file

with open(input_file,"r") as f:
    items=f.read().strip()

#Preapre the prompt 

prompt = f"""
You are an assistant that categorizes and sorts grocery items.
Here is a list of grocery items:
{items}
please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc. 
2. Sort the items in each category alphabetically.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.

"""
 
#send prompt and ge the response

try:
    response = ollama.generate(model=model,prompt=prompt)
    generated_text=response.get("response","")
    print("==== Categorized Grocery Items: ====\n")
    print(generated_text)

    with open(output_file,"w") as f:
        f.write(generated_text.strip())

    print(f"Successfully categorized the grocery items. The categorized list is saved in '{output_file}'")

except Exception as e:
    print(f"Error: {e}")
    exit(1)