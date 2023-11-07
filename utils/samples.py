few_shot_exp_samples = few_shot_samples = """
Question: Where does water in the sky come from?
Choices: ['1. space', '2. rain cloud', '3. surface of earth', '4. wishing well', '5. lake or river']
Answer: 5. lake or river
Explanation: The water in the lake or river will be evaporated by the sun and then form clouds.

Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
Choices: ['1. chess set', '2. strategy', '3. toy store', '4. chess game', '5. small case']
Answer: 1. chess set
Explanation: Because a chess set is a collection of chess pieces, it would be a good place to store a chess pawn when it is not being used. 

"""

few_shot_QA_samples = \
"""
Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
Choices: ['chess set', 'strategy', 'toy store', 'chess game', 'small case']
Answer: chess set

"""
fewshot_AE_samples = """
Question: Where does water in the sky come from?
Choices: ['1. space', '2. rain cloud', '3. surface of earth', '4. wishing well', '5. lake or river']
Answer: 5. lake or river
Explanation: The water in the lake or river will be evaporated by the sun and then form clouds.

Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
Choices: ['1. chess set', '2. strategy', '3. toy store', '4. chess game', '5. small case']
Answer: 1. chess set
Explanation: Because a chess set is a collection of chess pieces, it would be a good place to store a chess pawn when it is not being used. 

Please choose an answer and provide an explanation for the following questions in the same format as the above:

"""



# prompt for gpt-j to get a good answer, should be the same for everyone else
fewshot_QA2_samples = """
Question: Where does water in the sky come from?
Choices: space; rain cloud; surface of earth; wishing well; lake or river;
Answer: lake or river

Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
Choices: chess set; strategy; toy store; chess game; small case;
Answer: chess set

Please choose the correct answer for the following questions in the same format as the above:

"""

# Question: Where does water in the sky come from?
# Choices: 'space', 'rain cloud', 'surface of earth', 'wishing well', 'lake or river'
# Answer: lake or river

# Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
# Choices: 'chess set', 'strategy', 'toy store', 'chess game', 'small case'
# Answer: chess set

# Please choose the correct answer for the following questions in the same format as the above:


fewshot_QAE3_samples_new = """
Question: John cooled the steam. What did the steam become? Make a choice and provide an explanation.
Choices: 'condensate', 'electric smoke', 'smoke', 'liquid water', 'cold air'
The best choice is liquid water. When steam is cooled, it becomes liquid water.

Question: Where would you buy jeans in a place with a large number of indoor merchants? Make a choice and provide an explanation.
Choices: 'shopping mall', 'laundromat', 'hospital', 'clothing store', 'thrift store'
The best choice is shopping mall. Because a shopping mall is a place with a large number of indoor merchants and is a good place to buy jeans.

Question: I forgot to pay the electricity bill, now what can't I do with my ground pump? Make a choice and provide an explanation.
Choices: 'put in to the water', 'cause fire', 'produce heat', 'short fuse', 'shock'
The best choice is produce heat. This is because the ground pump can't produce heat without electricity.

"""

fewshot_QAE1_samples_new = """
Question: I forgot to pay the electricity bill, now what can't I do with my ground pump? Make a choice and provide an explanation.
Choices: 'put in to the water', 'cause fire', 'produce heat', 'short fuse', 'shock'
The best choice is 'produce heat'. This is because the ground pump can't produce heat without electricity.

"""


# fewshot_QA3_samples_new = """
# Question: John cooled the steam. What did the steam become?
# Choices: 'condensate', 'electric smoke', 'smoke', 'liquid water', 'cold air'
# Best choice: liquid water

# Question: Where would you buy jeans in a place with a large number of indoor merchants?
# Choices: 'shopping mall', 'laundromat', 'hospital', 'clothing store', 'thrift store'
# Best choice: shopping mall

# Question: I forgot to pay the electricity bill, now what can't I do with my ground pump?
# Choices: 'put in to the water', 'cause fire', 'produce heat', 'short fuse', 'shock'
# Best choice: produce heat

# """

# fewshot_QA3_samples_new = """
# Question: John cooled the steam. What did the steam become?
# Choices: 'condensate', 'electric smoke', 'smoke', 'liquid water', 'cold air'
# Answer: 'liquid water'

# Question: Where would you buy jeans in a place with a large number of indoor merchants?
# Choices: 'shopping mall', 'laundromat', 'hospital', 'clothing store', 'thrift store'
# Answer: 'shopping mall'

# Question: I forgot to pay the electricity bill, now what can't I do with my ground pump?
# Choices: 'put in to the water', 'cause fire', 'produce heat', 'short fuse', 'shock'
# Answer: 'produce heat'

# Please choose the correct answer for the following questions. Only the answer is required.

# """
# Zeroshot_QA_samples_new = """
# Choose the most appropriate one from Choices for this question based on commonsense:
# """
Zeroshot_QA_samples_new = """
Based on commonsense, please answer:
"""

falcon7b_Zeroshot_QA_samples_new = """
Answer this quesiton based on commonsense:
"""
# >>CONTEXT<<
# Based on commonsense, answer the question with only the correct choice:

mpt7b_instruct_Zeroshot_QA_samples_new = """
Based on commonsense:
"""

gpt_j_Zeroshot_QA_samples_new = """"""


Oneshot_QA_samples_new = """
Answer based on commonsense:

Question: I forgot to pay the electricity bill, now what can't I do with my ground pump?
Choices: 'put in to the water', 'cause fire', 'produce heat', 'short fuse', 'shock'
Answer: 'produce heat'

"""


# Please choose the correct answer for the following questions in the same format as the above:

fewshot_QAE2_samples = """

Question: Where does water in the sky come from?
Choices: 'space', 'rain cloud', 'surface of earth', 'wishing well', 'lake or river'
Answer: lake or river
Explanation: The water in the lake or river will be evaporated by the sun and then form clouds.

Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
Choices: 'chess set', 'strategy', 'toy store', 'chess game', 'small case'
Answer: chess set
Explanation: Because a chess set is a collection of chess pieces, it would be a good place to store a chess pawn when it is not being used. 

Please choose the correct answer for the following questions in the same format as the above:

"""

fewshot_AE2_samples = """
Question: Where does water in the sky come from?
Choices: 'space', 'rain cloud', 'surface of earth', 'wishing well', 'lake or river'
Answer: lake or river
Explanation: The water in the lake or river will be evaporated by the sun and then form clouds.

Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
Choices: 'chess set', 'strategy', 'toy store', 'chess game', 'small case'
Answer: chess set
Explanation: Because a chess set is a collection of chess pieces, it would be a good place to store a chess pawn when it is not being used. 

Please provide an explanation for the following questions in the same format as the above:

"""

# fewshot_QA2_samples = """
# Question: Where does water in the sky come from?
# Choices: space, rain cloud, surface of earth, wishing well, lake or river
# Answer: lake or river

# Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
# Choices: chess set, strategy, toy store, chess game, small case
# Answer: chess set

# Please choose the correct answer for the following questions in the same format as the above:

# """
# Please choose an answer for the following questions in the same format as the above:

# fewshot_QA2_samples = """
# Question: Where does water in the sky come from?
# Choices: ['1. space', '2. rain cloud', '3. surface of earth', '4. wishing well', '5. lake or river']
# Answer: 5. lake or river

# Question: If I wanted to store my chess pawn when I wasn't using it, what would be a good place for that?
# Choices: ['1. chess set', '2. strategy', '3. toy store', '4. chess game', '5. small case']
# Answer: 1. chess set

# Please choose an answer for the following questions in the same format as the above:

# """

# 
# Please provide an answer to the following questions like the above examples:
# Question: Where does water in the sky come from?
# Choices: ['space', 'rain cloud', 'surface of earth', 'wishing well', 'lake or river']
# Answer: lake or river

# few_shot_QA_samples = \
# """
# Copy the correct answer from the choices:
# """