import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import sys
import os
import pickle
import time
import csv

class FederatedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 7)
        self.conv2 = torch.nn.Conv2d(20, 40, 7)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.non_linearity = torch.nn.functional.relu

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            x = self.non_linearity(self.conv1(dummy_input))
            x = self.non_linearity(self.conv2(x))
            x = self.maxpool(x)
            flattened_size = x.view(1, -1).shape[1]

        self.linear = torch.nn.Linear(flattened_size, 10)

        self.track_layers = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'linear': self.linear
        }

    def forward(self, x):
        x = self.non_linearity(self.conv1(x))
        x = self.non_linearity(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def get_parameters(self):
        return {
            name: {
                'weight': layer.weight.data.clone(),
                'bias': layer.bias.data.clone()
            }
            for name, layer in self.track_layers.items()
        }

    def apply_parameters(self, parameters):
        if parameters is None:
            print("🛑 Nenhum parâmetro global fornecido. Treinando do zero.")
            return
        with torch.no_grad():
            for name in parameters:
                self.track_layers[name].weight.data.copy_(parameters[name]['weight'])
                self.track_layers[name].bias.data.copy_(parameters[name]['bias'])

class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        print(f"📦 Iniciando cliente {client_id}")
        start_time = time.time()
        self.dataset = self.load_data()
        print(f"📚 Dados carregados em {time.time() - start_time:.2f}s")

        # Criação do diretório de log
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/client_{client_id}_train.csv"
        with open(self.log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Batch", "Loss", "Tempo Batch (s)"])

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

        img, _ = full_dataset[0]
        print(f"🧪 DEBUG: Shape da imagem carregada: {img.shape} | Tipo: {img.dtype}")

        examples_per_client = len(full_dataset) // 3
        start_idx = self.client_id * examples_per_client
        end_idx = start_idx + examples_per_client
        indices = list(range(start_idx, end_idx))
        return Subset(full_dataset, indices)

    def train(self, parameters):
        net = FederatedNet()
        net.apply_parameters(parameters)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
        dataloader = DataLoader(self.dataset, batch_size=128, shuffle=True)

        print("🧠 Iniciando treinamento local...")
        start_time = time.time()

        for epoch in range(3):
            epoch_loss = 0.0
            batch_count = 0
            epoch_start = time.time()

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                batch_start = time.time()

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start
                epoch_loss += loss.item()
                batch_count += 1

                print(f"📊 Cliente {self.client_id} | Época {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f} | Tempo: {batch_time:.2f}s")

                with open(self.log_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([epoch + 1, batch_idx + 1, round(loss.item(), 4), round(batch_time, 2)])

            avg_loss = epoch_loss / batch_count
            epoch_time = time.time() - epoch_start
            print(f"📉 Cliente {self.client_id} | Época {epoch+1} concluída | Loss médio: {avg_loss:.4f} | Tempo época: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print(f"⏱️ Treinamento local finalizado em {total_time:.2f}s")

        conv1_w_mean = net.track_layers['conv1'].weight.data.mean().item()
        print(f"📈 Média final conv1.weight: {conv1_w_mean:.6f}")

        return net.get_parameters()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <parameters_path>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    parameters_path = sys.argv[2]

    print(f"🔧 Lendo parâmetros globais de {parameters_path}...")
    if os.path.exists(parameters_path):
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)
        print("✅ Parâmetros globais carregados.")
    else:
        print("⚠️ Parâmetros globais não encontrados. Treinando modelo do zero.")
        parameters = None

    client = Client(client_id)
    updated_parameters = client.train(parameters)

    output_path = f"client_{client_id}_parameters.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(updated_parameters, f)

    print(f"✅ Client {client_id} finalizou o treino. Parâmetros salvos em {output_path}")
