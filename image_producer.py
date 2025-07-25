from kafka import KafkaProducer
import time, os, json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

image_dir = '/Users/akshatmanral/Downloads/damaged-and-intact-packages'

for label in ['damaged', 'intact']:
    folder = os.path.join(image_dir, label)
    for img_file in os.listdir(folder):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            message = {'path': os.path.join(folder, img_file), 'label': label}
            try:
                record = producer.send('test_2', message)
                print(f"Sent: {message}")
                result = record.get(timeout=10)
                print(f"Delivered to: {result.topic}:{result.partition}")
                time.sleep(1)
            except Exception as e:
                print(f"Failed to send: {e}")

producer.flush()

