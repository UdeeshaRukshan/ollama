import ollama

response = ollama.list()

# == Chat example ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user", "content": "why is the ocean so salty?",
         },
    ],
    stream=True,
)
#print(res["message"]["content"])

for chunk in res:
    print(chunk["message"]["content"],end="",flush=True)


# == Generate example ==
res = ollama.generate(
    model="llama3.2",
    prompt="Tell me a short story and make it funny.",
)


# show
#print(ollama.show("llama3.2"))

modelfile = """
FROM llama3.2
SYSTEM You are very smart assistant who knows everything about ocean.You are very succinct and informative.
PARAMETER temperature 0.1
"""

ollama.create(model="knowitall", modelfile=modelfile)

res = ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
print(res["response"])

#delete model

ollama.delete("knowitall")