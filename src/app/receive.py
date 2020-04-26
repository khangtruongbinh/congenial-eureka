import pika

credentials = pika.PlainCredentials('antimatlab', 'antimatlab')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='10.10.10.102',  credentials=credentials))
channel = connection.channel()

channel.queue_declare(queue='abnormal_event')


def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)


channel.basic_consume(queue='abnormal_event', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()