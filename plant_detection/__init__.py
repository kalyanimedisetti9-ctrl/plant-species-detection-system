# PyMySQL wrapper to allow MySQL database connection via Django default MySQL backend
try:
    import pymysql
    pymysql.install_as_MySQLdb()
except ImportError:
    pass
