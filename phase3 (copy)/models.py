class User:
	"""get user data from table User"""
	def __init__(self, db=None):
		self.db = db

	def get_user(self, username):
		"""Read user infomation through {username}"""
		cur = self.db.connection.cursor()
		query = "SELECT userId, username, name, password FROM User WHERE username = '{username}' ".format(username=username)
		cur.execute(query)

		rows = cur.fetchall()
		cur.close()

		#print rows
		return rows
