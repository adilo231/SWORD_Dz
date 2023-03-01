from Transformer.Could_of_word import CloudOfWord

mongo_db = "twitter_db"
mongo_user = "Prix"
mongo_uri = "mongodb://localhost:27017/"

# cloud =CloudOfWord(mongo_uri,mongo_db,mongo_user,lang='french')
# cloud.print_Could()
def Algerian_location(location):
    Locations=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','Algiers, Algeria']
    if location in Locations:
        return True
    for loc in Locations:
        if loc in location: return True
    return False





print(Algerian_location('Alger'))
print(Algerian_location('skikda_Alger'))
print(Algerian_location('skikda - Algérie'))
print(Algerian_location('الجزائر - hsh'))
print(Algerian_location('us - hsh'))