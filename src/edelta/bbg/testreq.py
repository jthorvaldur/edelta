import blpapi

options = blpapi.SessionOptions()
options.setServerHost("localhost")
# options.setServerPort(8194)
session = blpapi.Session(options)
session.start()
