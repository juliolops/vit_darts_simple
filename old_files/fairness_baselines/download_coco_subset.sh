#!/usr/bin/env bash
set -e  # Sale si hay algún error

# 1. Crear estructura de carpetas
BASE_DIR="datasets/COCO_sub"
ANN_DIR="$BASE_DIR/annotations"
IMG_DIR_TRAIN="$BASE_DIR/train2017"
IMG_DIR_VAL="$BASE_DIR/val2017"
OUT_DIR="$BASE_DIR/cocobin_data"

mkdir -p "$ANN_DIR" "$IMG_DIR_TRAIN" "$IMG_DIR_VAL"

# Función genérica de extracción por Python
extract_with_python() {
  ZIP_PATH="$1"
  DEST_DIR="$2"
  shift 2
  MEMBERS=("$@")  # lista de miembros (relativos dentro del ZIP); si vacía, extrae todo

  python3 - <<PYCODE
import zipfile, os
zip_path = os.path.expanduser("$ZIP_PATH")
dest_dir = os.path.expanduser("$DEST_DIR")
os.makedirs(dest_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    if ${#MEMBERS[@]} > 0:
        for member in ${MEMBERS[@]+"${MEMBERS[@]}"}:
            target = os.path.join(dest_dir, os.path.basename(member))
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    else:
        z.extractall(dest_dir)
PYCODE
}

# 1. Anotaciones
ZIP_ANN="$ANN_DIR/annotations_trainval2017.zip"
if [ -f "$ZIP_ANN" ]; then
  echo "→ $ZIP_ANN ya existe, omitiendo descarga."
else
  echo "Descargando anotaciones COCO..."
  wget -q -O "$ZIP_ANN" \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

echo "Extrayendo sólo JSON de anotaciones..."
if command -v unzip >/dev/null; then
  unzip -q -j "$ZIP_ANN" \
    'annotations/instances_train2017.json' \
    'annotations/instances_val2017.json' \
    -d "$ANN_DIR"
else
  extract_with_python \
    "$ZIP_ANN" \
    "$ANN_DIR" \
    "annotations/instances_train2017.json" \
    "annotations/instances_val2017.json"
fi

# 2. Imágenes de entrenamiento
ZIP_TRAIN="$BASE_DIR/train2017.zip"
if [ -f "$ZIP_TRAIN" ]; then
  echo "→ $ZIP_TRAIN ya existe, omitiendo descarga."
else
  echo "Descargando train2017.zip..."
  wget -c -q -P "$BASE_DIR" http://images.cocodataset.org/zips/train2017.zip
fi

echo "Descomprimiendo train2017..."
if command -v unzip >/dev/null; then
  unzip -q "$ZIP_TRAIN" -d "$BASE_DIR"
else
  extract_with_python "$ZIP_TRAIN" "$BASE_DIR"
fi

# 3. Imágenes de validación
ZIP_VAL="$BASE_DIR/val2017.zip"
if [ -f "$ZIP_VAL" ]; then
  echo "→ $ZIP_VAL ya existe, omitiendo descarga."
else
  echo "Descargando val2017.zip..."
  wget -c -q -P "$BASE_DIR" http://images.cocodataset.org/zips/val2017.zip
fi

echo "Descomprimiendo val2017..."
if command -v unzip >/dev/null; then
  unzip -q "$ZIP_VAL" -d "$BASE_DIR"
else
  extract_with_python "$ZIP_VAL" "$BASE_DIR"
fi

# 4. Filtrado y balanceo
#echo "Ejecutando build_coco_binary.py…"
#python build_coco_binary.py

echo "¡Todo listo! 🙌"