import os
from run_evolution_moqnas import main  # Importamos la función main de tu script MO-QNAS

# Si necesitas definir variables de entorno (por ejemplo, GPUs), hazlo aquí
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parámetros para la ejecución de prueba (ajusta rutas y nombres según tu estructura)
parameters = {
    "experiment_path": "experimento_moqnas_cifar10/exp1_repeat_2",
    "data_path": "cifar10_data",       # Ajusta a la ruta real de tu dataset
    "dataset": "cifar10",
    "config_file": "config_files_cifar/config0_0.txt",  # Debe contener MOQNAS_spec
    "continue_path": "",       # Si quisieras retomar una ejecución previa, pon aquí la carpeta
    "log_level": "DEBUG",      # DEBUG para salida detallada en pruebas
    "optimizer": "AdamW",
    "fitness_metric": "best_accuracy",
    "data_augmentation": False,
    "early_stopping": False,
    "en_pop_crossover": True,
    "save_checkpoints_epochs": 5,
    "limit_data_value": 10000,
    "backbone_name": "resnet18",
    "network_config": "default",
    "multi_objective": True,
    
}

if __name__ == "__main__":
    # Llama a main(...) con los parámetros definidos arriba
    main(**parameters)
