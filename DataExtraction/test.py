#### to test fb extractor

query="ramadan"
max_posts=10
type_target="search"#type_target="profil"
type_data="posts"  #type_data="posts"      #comments=posts+comments
fb=FacebookExtractor(fb.driver)
fb.search(query,type_target,type_data)
fb.posts
