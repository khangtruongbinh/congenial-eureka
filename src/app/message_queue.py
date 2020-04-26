import pika
import json
import socket
from datetime import datetime
from config.config import get_config

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

message_channel = get_config('app', 'message_channel')
channel.queue_declare(queue=message_channel)

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
timestamp = 1572175892
data = {
    'event_url': '%s:%s/event/%s'%(IPAddr, get_config('app', 'port'), timestamp),
    'cam_id': 'B1-L101',
    'time': datetime.now().strftime('%d/%m/%Y, %H:%M:%S')
}
message = json.dumps(data)

channel.basic_publish(exchange='', routing_key=message_channel, body=message)
print(" [x] Sent message")
connection.close()