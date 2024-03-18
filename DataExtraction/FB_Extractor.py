from selenium import webdriver
from bs4 import BeautifulSoup
import facebook_scraper as fs
from time import sleep
import re
from time import sleep
import json
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
  

class FacebookExtractor():

    def Login(self):
        self.driver.get("https://www.facebook.com")

        # Find the email and password input fields
        # Find the email field using the ID
        email_field = self.driver.find_element(webdriver.common.by.By.ID, "email")
        password_field = self.driver.find_element(webdriver.common.by.By.ID, "pass")

        # Enter the email and password
        email_field.send_keys(self.username)
        password_field.send_keys(self.password)

        # Find the login button
        login_button = self.driver.find_element(webdriver.common.by.By.NAME, "login")

        # Click the login button
        login_button.click()
    


    def __init__(self,Facebook_username=None,Facebook_password=None):
        chrome_options = webdriver.ChromeOptions()
        #make browser hidden through the options
        self.username=Facebook_username
        self.password=Facebook_password
        self.driver = driver
        webdriver.Chrome(options=chrome_options)
        self.posts=[]
        if Facebook_username == None or Facebook_password == None :
            # Open the Twitter login page
            f = open('authentification.json')
            data= json.load(f)
            Facebook_data=data['Facebook']
            self.username=Facebook_data['username']
            self.password=Facebook_data['password']
        self.posts=[]
        self.Login()

    def view_posts(self):
        return self.posts

    def get_id_type(self, url):
        post_id = None
        post_type = None

        # Prioritize specific patterns to avoid potential conflicts
        if "/videos/pcb." in url:
            match = re.search(r"/videos/pcb.([a-zA-Z0-9]+)", url)
            if match:
                post_id = match.group(1)
                post_type = "video_playlist"  # Assuming "pcb" refers to video playlist
            else:
                raise ValueError(f"Invalid URL format for video playlist: {url}")
        elif "/photos/a." in url:
            match = re.search(r"/photos/a.([a-zA-Z0-9]+)", url)
            if match:
                post_id = match.group(1)
                post_type = "album"  # Assuming "a" refers to album
            else:
                raise ValueError(f"Invalid URL format for album: {url}")
        elif "/posts/" in url:
            match = re.search(r"/posts/([a-zA-Z0-9]+)", url)
            if match:
                post_id = match.group(1)
                post_type = "post"
            else:
                raise ValueError(f"Invalid URL format for post: {url}")

        # Less specific patterns (consider ordering carefully if conflicts arise)
        elif "/videos/" in url:
            post_type = "video"  # Generic video case
        elif "permalink.php" in url:
            # Extract ID from 'story_fbid' parameter
            match = re.search(r"story_fbid=([a-zA-Z0-9]+)", url)
            if match:
                post_id = match.group(1)
                post_type = "post"  # Assuming 'permalink.php' indicates a post
            else:
                raise ValueError(f"Invalid URL format for permalink.php: {url}")

        return post_id, post_type



    def get_post(self,post_id,MAX_COMMENTS): #responsible for retrieving post data and can also get the post with it's comments.
        post_data = []
        gen = fs.get_posts(post_urls=[post_id], options={"comments": MAX_COMMENTS, "progress": False})
        post = next(gen)
        comments = post.get('comments_full', [])
        content = post.get('text', [])
        post_data.append({
        'id': post_id,
        'text': content,
        'comments': comments
        })
        return post_data



    def search(self,query,type_target,type_data,max_numbers_posts=1,max_numbers_comments=0):
        #query can be a : profile name or page,keyword,post_id
        #max_numbers : is the maximum number of posts or comments
        #type_target : to see if it's a profile ,or a keyword,or an id comment (we use it to adjust the link)
        #type_data : if posts or comments+posts
        if type_data != "posts" :
          max_numbers_comments=10000000
        if type_target=="search": #search using a keyword
            pageFB = f"https://www.facebook.com/search/top?q={query}"
        
        elif type_target=="post_id" : #get comments of a given post id    (akhdmha b lien)
            self.posts = self.get_post(query,max_numbers_comments)
            return
        else :  #profile
            pageFB = f"https://www.facebook.com/{query}"
            
        self.driver.get(pageFB)
        
        post_ids=[]
        previous_height=0
        scroll = True
        while len(post_ids) < int(max_numbers_posts) and scroll :
            try:

                self.driver.implicitly_wait(10)
                current_height = self.driver.execute_script("return document.body.scrollHeight")


                if current_height == previous_height:
                    scroll = False
                    return
                else:
                    previous_height = current_height
                

                posts = self.driver.find_elements(By.XPATH, ".//div[contains(@role, 'article')]")

                for post in posts :
                    a_elements = post.find_elements(By.TAG_NAME, "a")
                    for a_element in a_elements :
                        link = a_element.get_attribute("href")
                        post_id , post_type =self.get_id_type(link)
                        if post_id != None and post_type == "post" : 
                            #check if id already exists
                            if post_id not in post_ids:
                                post_ids.append(post_id)

                    if scroll:
                        try:
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        except Exception as e:
                            print(f"Error scrolling: {e}")  # Log any scrolling errors
                            scroll = False
            except :
                   return


        posts = []
        for id in post_ids:
            post=self.get_post(id,max_numbers_comments)
            posts.append(post)
        self.posts=posts


