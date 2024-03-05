list_of_tokens = ['word1','word1','word2']
language = 'english'

cloud_generator = CloudOfWords()
word_cloud = cloud_generator.generateCW(list_of_tokens, language)

cloud_generator.showCW()
