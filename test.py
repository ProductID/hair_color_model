# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``
#
# import requests
#
# model_inputs = {'prompt': 'Hello I am a [MASK] model.'}
#
# res = requests.post('http://localhost:8000/', json = model_inputs)
#
# print(res.json())

import pysftp

srv = pysftp.Connection(host="3.70.151.70", username="ubuntu",password="sd9809$%^")


#  upload file
with srv.cd('/home/ubuntu/uploadimages'):  # chdir to public
    srv.put("/home/rnative/Downloads/christopher-campbell-rDEOVtE7vOs-unsplash.jpg")  # upload file to nodejs/

# download file
srv.get("/home/ubuntu/uploadimages/christopher-campbell-rDEOVtE7vOs-unsplash.jpg","christopher-campbell-rDEOVtE7vOs-unsplash.jpg")

# Closes the connection
srv.close()