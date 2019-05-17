import xnmt

file_manager = {}

def request_text_file(filename):
  global file_manager
  if filename not in file_manager:
    xnmt.logger.info(f">> Begin reading: {filename}")
    with open(filename, encoding='utf-8') as f:
      file_manager[filename] = f.readlines()
    xnmt.logger.info(f">> Finished reading: {filename}")
  return file_manager[filename]

def clear_cache():
  file_manager.clear()
