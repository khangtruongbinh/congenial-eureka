import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def get_tokens():
    # Use a service account
    cred = credentials.Certificate('hackathonalertapp-a9c1ea8b1a9f.json')
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    # [START quickstart_get_collection]
    users_ref = db.collection(u'antimatlab')
    docs = users_ref.stream()
    res = [doc.to_dict()['token'] for doc in docs]
    return res
