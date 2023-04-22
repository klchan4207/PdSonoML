import os
import json

backend_name = 'tensorflow'

# to avoid import error at dgl/backend/__init__.py
default_dir = None
if "DGLDEFAULTDIR" in os.environ:
    default_dir = os.getenv("DGLDEFAULTDIR")
else:
    default_dir = os.path.join(os.path.expanduser("~"), ".dgl")
config_path = os.path.join(default_dir, "config.json")
with open(config_path,'r') as input:
    _config = json.loads(input.read())
    _config['backend'] = backend_name
with open(config_path,'w+') as output:
    output.write(json.dumps(_config))