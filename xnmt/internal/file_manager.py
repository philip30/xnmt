file_manager = {}

def request_text_file(filename):
  global file_manager
  if filename not in file_manager:
    with open(filename, encoding='utf-8') as f:
      file_manager[filename] = f.readlines()
  return file_manager[filename]

def clear_cache():
  file_manager.clear()