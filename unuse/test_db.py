from unuse import test_func as ts
from util import save_2_db as db

ts.testFunc()
from unuse import create_unique as tags

tag=tags.createUnitag()
db.store_single('1', '2', '3', '4', 1, tag)

db.store_total('1', '2', '3', 4, 5, tag)