import time
import torch
import torch.nn as nn
# Se elimina la importación de torchvision.models porque ya no se usa
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import medmnist
from medmnist import INFO
import os
import json

# --- 1. IMPORTAR TUS MODELOS PERSONALIZADOS ---
# Asumiendo la estructura de carpetas recomendada
from cnn.model_resnet import ResNet18, ResNet50

def calculate_mean_std(loader: DataLoader):
    """
    Calcula la media y la desviación estándar de un dataset de forma iterativa.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for images, _ in loader:
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    return mean, std

class InferenceRunner:
    """
    Clase que encapsula un modelo de PyTorch para medir su tiempo de inferencia.
    """
    # ... (El código de esta clase no cambia)
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self._to_device(d) for d in data]
        return data.to(self.device)
    def _check_device(self, data):
        if isinstance(data, (list, tuple)):
            return all(d.device == torch.device(self.device) for d in data)
        return data.device == torch.device(self.device)
    def measure_inference_time(self, input_data, warmup_runs=10, measure_runs=10):
            """
            Mide el tiempo de inferencia y devuelve la media y la desviación estándar.
            """
            self.model.eval()
            if not self._check_device(input_data):
                input_data = self._to_device(input_data)
            
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(input_data)
            
            inference_times = []
            with torch.no_grad():
                for _ in range(measure_runs):
                    if 'cuda' in self.device and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    _ = self.model(input_data)
                    if 'cuda' in self.device and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    inference_times.append(end_time - start_time)
            
            # Convertir a microsegundos
            inference_times_us = [t * 1e6 for t in inference_times]
            
            # Calcular media
            avg_time_us = sum(inference_times_us) / len(inference_times_us)
            
            # Calcular desviación estándar
            if len(inference_times_us) > 1:
                std_time_us = torch.std(torch.tensor(inference_times_us)).item()
            else:
                std_time_us = 0.0

            return avg_time_us, std_time_us



if __name__ == '__main__':
    MODEL_PATHS = {
        'resnet18': {
            'pathmnist': 'medmodels/pathmnits/resnet18_on_pathmnist.pth',
            'tissuemnist': 'medmodels/tissuemnist/resnet18_on_tissuemnist.pth',
            'organamnist': 'medmodels/organamnist/resnet18_on_organamnist.pth'
        },
        'resnet50': {
            'pathmnist': 'medmodels/pathmnits/resnet50_on_pathmnist.pth',
            'tissuemnist': 'medmodels/tissuemnist/resnet50_on_tissuemnist.pth',
            'organamnist': 'medmodels/organamnist/resnet50_on_organamnist.pth'
        }
    }
    DATASETS_TO_TEST = ['pathmnist', 'tissuemnist', 'organamnist']
    MODELS_TO_TEST = ['resnet18', 'resnet50']
    BATCH_SIZE = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    print(f"🚀 Iniciando pruebas de inferencia en el dispositivo: {DEVICE.upper()}")
    print("-" * 60)

    for model_name in MODELS_TO_TEST:
        for dataset_name in DATASETS_TO_TEST:
            print(f"🔬 Probando modelo '{model_name}' en dataset '{dataset_name}'...")
            model_path = MODEL_PATHS[model_name][dataset_name]

            if not os.path.exists(model_path):
                print(f"   🚨 Error: No se encontró el archivo de pesos en '{model_path}'. Saltando esta prueba.\n")
                continue

            info = INFO[dataset_name]
            n_classes = len(info['label'])
            DataClass = getattr(medmnist, info['python_class'])
            print("   -> Calculando media y desviación estándar...")
            temp_train_dataset = DataClass(split='train', download=True, as_rgb=True, transform=ToTensor())
            temp_train_loader = DataLoader(dataset=temp_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            mean, std = calculate_mean_std(temp_train_loader)
            print(f"   -> Media: {mean.tolist()}, Std: {std.tolist()}\n")
            test_transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
            test_dataset = DataClass(split='test', download=True, as_rgb=True, transform=test_transform)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            images, _ = next(iter(test_loader))
            input_data = images[:10]

            # --- 2. CARGAR EL MODELO USANDO TU DEFINICIÓN ---
            try:
                # Se usa in_channels=3 porque as_rgb=True convierte las imágenes a 3 canales
                if model_name == 'resnet18':
                    model = ResNet18(in_channels=3, num_classes=n_classes)
                elif model_name == 'resnet50':
                    model = ResNet50(in_channels=3, num_classes=n_classes)
                else:
                    # Si añadieras más modelos, se podrían manejar aquí
                    raise ValueError(f"Modelo '{model_name}' no reconocido.")

                # Cargar los pesos. Ahora debería funcionar sin errores.
                state_dict = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=True)
                model.load_state_dict(state_dict)
                print(f"   -> Pesos para '{model_name}/{dataset_name}' cargados desde '{model_path}'")
            except Exception as e:
                print(f"   🚨 Error al cargar el modelo '{model_name}': {e}. Saltando esta prueba.\n")
                continue

            # --- Medición y guardado de resultados (sin cambios) ---
            runner = InferenceRunner(model, device=DEVICE)
            test_batch_size = input_data.shape[0]
            avg_time_us, std_time_us = runner.measure_inference_time(input_data)
            print(f"   ✅ Tiempo de inferencia promedio: {avg_time_us:.2f} ± {std_time_us:.2f} µs por lote de {test_batch_size} imágenes.\n")

            result_entry = {
                'model': model_name,
                'dataset': dataset_name,
                'weights_file': os.path.basename(model_path),
                'device': DEVICE,
                'batch_size_tested': test_batch_size,
                'average_inference_time_us': round(avg_time_us, 3),
                'std_dev_inference_time_us': round(std_time_us, 3)
            }
            all_results.append(result_entry)

    results_filename = 'inference_results.json'
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"🏁 Pruebas finalizadas. Resultados guardados en '{results_filename}'.")