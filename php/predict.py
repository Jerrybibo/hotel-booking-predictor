import sys
import base64
import json

arg = sys.argv[1]
content = json.loads(base64.b64decode(arg))

print(content)
