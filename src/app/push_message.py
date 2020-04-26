from pyfcm import FCMNotification
from config.config import get_config
import socket
from get_token import get_tokens
from datetime import datetime
import requests
import json


push_service = FCMNotification(api_key="AIzaSyBi-FwrNCZefPqNtz7SjLe9k66i8ndwQcY")
# Your api-key can be gotten from:  https://console.firebase.google.com/project/<project-name>/settings/cloudmessaging


def push_message(timestamp, location='B1-TMB101', sound=True):
    # Send to multiple devices by passing a list of ids.
    global push_service
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    data_message = {
        'event_url': '%s:%s/event/%s'%(IPAddr, get_config('app', 'port'), timestamp)
    }
    # 'icon_url': 'http://antimatlab.com/img/logo.png'
    print(data_message)
    registration_ids = get_tokens()
    message_title = "CẢNH BÁO BẤT THƯỜNG!"
    message_body = "Cam: {} Time: {}".format(location, datetime.fromtimestamp(timestamp))
    result = push_service.notify_multiple_devices(registration_ids=registration_ids, message_title=message_title,
                                                  message_body=message_body, sound=sound, data_message=data_message)
    print(result)

def push_firestore(timestamp, location):
    url = 'https://delta-entry-160518.firebaseio.com/controllers.json'
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    data = {
        'timestamp': '{}'.format(timestamp).replace('.', '')[:-3],
        'cam_id': 'CamID: {}'.format(location),
        'datetime': 'Time: {}'.format(datetime.fromtimestamp(int(timestamp))),
        # 'event_url': 'http://%s:%s/event/%s' % (IPAddr, get_config('app', 'port'), int(timestamp))
        'event_url': 'http://{}:{}/event/1572175892'.format(IPAddr, get_config('app', 'port'))
    }
    headers = {"Content-Type": "application/json"}
    response = requests.put(url, data=json.dumps(data), headers=headers)
    res = response.json()
    print(res)

if __name__=='__main__':
    # Push firebase
    # push_message(sound=False, timestamp=1572175892)
    ts = datetime.now().timestamp()
    # Push firestore 157.522.0234.131
    push_firestore(timestamp=ts, location='A2-HL25A')