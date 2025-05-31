# 🍇 Verificador Inteligente de Hojas de Uva

Una API desarrollada con **Python**, **TensorFlow** y **FastAPI** que utiliza redes neuronales convolucionales para detectar si una imagen contiene hojas de uva de manera automática y precisa.

## 🌟 Características Principales

- **🔍 Detección Automática**: Identifica si la imagen tiene o no hojas de uva
- **🎯 Alta Precisión**: Modelo optimizado con Keras Tuner

## 🚀 Instalación Rápida

### Prerrequisitos

- **Python 3.12+** (recomendado 3.12.3 o superior)
- **FFmpeg** (para generación de videos)

### 1. Clonar o Descargar el Proyecto

```bash
# Si tienes git instalado
git clone https://github.com/HarDep/grape-leaf-verificator.git
cd grape-leaf-verificator

# O descarga los archivos y colócalos en una carpeta
```

### 2. Instalar Dependencias

**Opción A: Instalación manual**
```bash
pip install -r requirements.txt
```

**Opción B: Instalación individual**
```bash
pip install tensorflow opencv-python Pillow numpy ffmpeg-python
```

### 3. Instalar FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- Descargar desde [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Agregar al PATH del sistema

## 🎬 Uso de la Aplicación

### Iniciar la Aplicación

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Acceder a la Interfaz Web

1. Abre tu navegador web
2. Ve a: `http://localhost:8000/docs`
3. ¡La aplicación estará lista para usar!

### Cómo Usar

1. **📸 Subir Imagen**: Usa el endpoind
2. **🔍 Analizar**: Selecciona la imagen
3. **📋 Ver Resultados**: Revisa si la imagen tiene o no hojas de uva

## 🔧 Solución de Problemas

### Error con FFmpeg
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Descargar e instalar desde el sitio oficial

### Reportar Errores
1. Ve a la sección de Issues
2. Describe el problema detalladamente
3. Incluye capturas de pantalla si es posible

### Mejoras Propuestas
- 🔄 Soporte para más formatos de imagen
- 📱 Aplicación móvil nativa
- 🌍 Soporte multiidioma
- 📈 Métricas de rendimiento en tiempo real
- 🗂️ Historial de análisis

## 👨‍💻 Desarrollador

Desarrollado con ❤️ usando:
- **TensorFlow** para el modelo de IA
- **Gradio** para la interfaz web
- **OpenCV** para procesamiento de imágenes
- **Matplotlib** para visualizaciones

---

*¿Tienes preguntas o necesitas ayuda? No dudes en contactarnos o crear un Issue en el repositorio.*