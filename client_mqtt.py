import paho.mqtt.client as mqtt
import pickle
import sys
import os
import subprocess
import threading
import time
import csv

# ========= Configuração do MQTT =========

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("📡 Conectado ao broker MQTT.")
        client_id = userdata['client_id']
        topic = f"server/global_parameters/{client_id}"
        client.subscribe(topic)
        print(f"📥 Subscrito em {topic}")
    else:
        print(f"❌ Falha na conexão com o broker. Código de retorno: {rc}")

def on_message(client, userdata, msg):
    topic = msg.topic
    client_id = userdata['client_id']

    print(f"📥 Parâmetros globais recebidos para cliente {client_id}.")
    start_unpack = time.time()
    global_parameters = pickle.loads(msg.payload)
    unpack_time = time.time() - start_unpack
    print(f"📦 Parâmetros desembrulhados em {unpack_time:.2f}s (tamanho: {len(msg.payload)} bytes)")

    userdata['parameters'] = global_parameters
    userdata['unpack_time'] = unpack_time
    userdata['recv_size'] = len(msg.payload)
    userdata['received'].set()

# ========== Execução principal ==========

if __name__ == "__main__":
    total_start = time.time()

    if len(sys.argv) != 3:
        print("Uso: python client_mqtt.py <client_id> <broker_ip>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    broker_ip = sys.argv[2]

    print(f"🚀 Iniciando cliente MQTT {client_id} | Broker: {broker_ip}")

    userdata = {
        'client_id': client_id,
        'parameters': None,
        'received': threading.Event(),
        'unpack_time': 0,
        'recv_size': 0
    }

    mqtt_start = time.time()
    client = mqtt.Client(userdata=userdata, protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_ip, 1883, 60)
    client.loop_start()
    print("⏳ Aguardando parâmetros globais...")

    userdata['received'].wait()
    mqtt_total = time.time() - mqtt_start
    print(f"📡 Tempo total aguardando parâmetros via MQTT: {mqtt_total:.2f}s")

    print("🚀 Iniciando subprocesso de treinamento com client.py...")
    train_start = time.time()

    temp_param_file = f"temp_global_params_client_{client_id}.pkl"
    with open(temp_param_file, 'wb') as f:
        pickle.dump(userdata['parameters'], f)

    result = subprocess.run([
        sys.executable,
        "client.py",
        str(client_id),
        temp_param_file
    ])

    train_duration = time.time() - train_start
    print(f"✅ Subprocesso de treinamento finalizado em {train_duration:.2f}s")

    updated_param_file = f"client_{client_id}_parameters.pkl"
    if os.path.exists(updated_param_file):
        with open(updated_param_file, 'rb') as f:
            updated_params = pickle.load(f)

        payload = pickle.dumps(updated_params)
        topic = f"client/params/{client_id}"
        client.publish(topic, payload)
        send_size = len(payload)
        print(f"📤 Parâmetros locais publicados em {topic} ({send_size} bytes)")
    else:
        print(f"⚠️ Arquivo {updated_param_file} não encontrado. Verifique se o client.py executou corretamente.")
        send_size = 0

    client.loop_stop()
    client.disconnect()

    total_end = time.time() - total_start
    print(f"🕓 Tempo total de execução do client_mqtt.py: {total_end:.2f}s")

    # ========= Salvar métricas em CSV =========
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"client_{client_id}_metrics.csv")

    with open(log_path, mode="w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "client_id", "mqtt_time", "unpack_time", "recv_bytes",
            "train_time", "send_bytes", "total_time"
        ])
        writer.writerow([
            client_id, round(mqtt_total, 4), round(userdata['unpack_time'], 4),
            userdata['recv_size'], round(train_duration, 4),
            send_size, round(total_end, 4)
        ])

    print(f"📝 Métricas salvas em {log_path}")
