import os,jinja2
 
# Statement for enabling the development environment
DEBUG = True
 
# Define the application directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
 
# Application threads. A common general assumption is using 2 per available processor cores 
# to handle incoming requests using one and performing background operations using the other.
THREADS_PER_PAGE = 2
 
# Enable protection agains *Cross-site Request Forgery (CSRF)*
CSRF_ENABLED = True
 
# Use a secure, unique and absolutely secret key for signing the data.
CSRF_SESSION_KEY = "18uejhs7H8r65GFR5fTfd5HjTR"
 
# Secret key for signing cookies
SECRET_KEY = "3R42Rd3f56q46fgDRrt85R5R7Ryu"
 
# TEMPLATES FOLDER
TEMPLATE_FOLDER = "templates"
 
# JINJA LOADER
JINJA_ENVIRONMENT = jinja2.ChoiceLoader([
    jinja2.FileSystemLoader([x[0] for x in os.walk('templates')]),
])
 
# ASSETS FOLDER
ASSETS_FOLDER = "style"
 
# PHPMYADMIN DB CONFIGURATION
#"""
MYSQL_HOST = 'sql3.freesqldatabase.com'
MYSQL_USER = 'sql3243349'
MYSQL_PASSWORD = 'KqrZtxHXpP'
MYSQL_DB = 'sql3243349'
#"""

# LOCAL DB CONFIGURATION
"""
MYSQL_HOST = 'localhost'
MYSQL_USER = 'gatechUser'
MYSQL_PASSWORD = 'gatech123'
MYSQL_DB = 'cs6400_summer18_team032'
"""
