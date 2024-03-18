#### to test fb extractor

query="ramadan"
max_posts=10
type_target="search"
type_data="posts"
num_comments=0
fb=FacebookExtractor()
fb.search(query,type_target,type_data,max_posts,num_comments)
print(fb.view_posts())
