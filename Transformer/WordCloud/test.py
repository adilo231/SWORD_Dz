language = 'ar'

cloud_generator = CloudOfWords("Twitter_Collections","Algeria")
cloud_generator.generateCW(language,r'./NotoSansArabic_SemiCondensed-ExtraBold.ttf',100)
cloud_generator.showCW()
