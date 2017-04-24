import time

def create_unique():
    tag=time.time()*1000
    tag=round(tag)
    return str(tag)
