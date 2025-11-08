import pymongo, certifi
uri = "mongodb+srv://dbflaskuser:flask123@ac-mk8l0qk-shard-00-00.tvpcrxu.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10000)
try:
    info = client.server_info()
    print("✅ Connected successfully!")
    print(info)
except Exception as e:
    import traceback; traceback.print_exc()
