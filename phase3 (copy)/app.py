from flask import Flask, render_template, url_for, request, redirect, session, flash, g, make_response, send_file
#from flaskext.mysql import MySQL
from flask_mysqldb import MySQL
#from flask_security import Security, login_required
#from wtforms import Form, BooleanField, TextField, PasswordField, validators
from functools import wraps
import gc

from config import *

#import classes
from models import User

app = Flask(__name__, static_folder=ASSETS_FOLDER, template_folder=TEMPLATE_FOLDER)

app.secret_key = SECRET_KEY

app.config.from_object('config')

mysql = MySQL(app)
#mysql.init_app(app)

#initialize objects
users = User(db=mysql)

#Home page
@app.route('/')
def homepage():
	return render_template("index.html")

def login_required(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args, **kwargs)
		else:
			flash("Please login first.")
			return redirect(url_for('login'))

	return wrap

#User login with username and password
@app.route('/login/', methods=['GET','POST'])
def login():
	try:
		error = None
		if request.method == 'POST':

			data = format_users(users.get_user(request.form['username']))[0]
			pw = data['password']
			print "data: ", data

			if pw == request.form['password']:
				session['logged_in'] = True
				session['username'] = request.form['username']

				#print "username: ", session['username']
				#print "password: ", pw

				return redirect(url_for('show_user', username=data['username']))

			else:
				error = 'Your username or password is not corect. Please try again.'

		gc.collect()

		#return render_template('login.html', error=error)
		return redirect(url_for('show_user', username=data['username']))

	except Exception, e:
		print "exception: ", e
		error = 'Invalid credentials. Try again'

		return render_template('login.html', error=error)

#User logout
@app.route('/logout')
@login_required
def logout():
	session.clear()
	#flash("You have been logged out.")
	gc.collect()

	return redirect(url_for('homepage'))

#User <username>'s dashboard
@app.route('/users/<username>', methods=['GET', 'POST'])
@login_required
def show_user(username):
    user=format_users(users.get_user(username))[0]
    #print "user data: ", user

    return render_template('show_user.html', user = user)


#Helper functions
def format_users(data):
	users = []

	for row in data:
		users.append ({
			'id':row[0],
			'username': row[1],
			'name': row[2],
			'password': row[3]
		})

	return users

if __name__ == '__main__':
	app.debug = True
	app = run()
