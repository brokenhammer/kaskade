global_records = None

def _init():
    global global_records
    global_records = []

def _append(obj):
    global global_records
    global_records.append(obj)

def _clear():
    global global_records
    global_records.clear()

def _get():
    global global_records
    return global_records