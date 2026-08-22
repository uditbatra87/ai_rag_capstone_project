import tiktoken

text = """Nature is a wonderful gift that gives us clean air, fresh water, and green trees. 
Trees are very important for our world because they provide oxygen, shelter, and food for all living creatures. 
When we protect the environment, we help animals and plants survive and grow. 
Sadly, many people cut down forests for buildings and roads, which harms the earth. 
We must plant more trees, stop wasting water, and keep our surroundings clean. 
Every small action helps to make our planet a better, 
greener, and safer place for future generations to live a happy and healthy life. """


encoder = tiktoken.encoding_for_model('gpt-4o')
tokens = encoder.encode(text)

print(f"Length of tokens is  : {len(tokens)}")
print(tokens)

# cost = (prompt_tokens x in_rate) + (completion_tokens x out_rate)

