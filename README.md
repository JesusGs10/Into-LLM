## Introducción a los LLM

Este repositorio documenta el desarrollo e implementación progresiva de varios modelos de lenguaje, desde los conceptos fundamentales hasta arquitecturas modernas como los Transformers. El objetivo principal es construir y entender modelos generativos de texto a nivel de carácter, explorando la evolución de las técnicas y arquitecturas, basándonos en la serie de tutoriales **Neural Networks: Zero to Hero** de **Andrej Karpathy** https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ.

### Contenido Detallado del Proyecto:

El proyecto se estructura en los siguientes grandes bloques:

1.  **Introducción a las Redes Neuronales:**
    *   Un repaso histórico desde los pioneros (McCulloch-Pitts, Hebb) hasta modelos como Hopfield, Máquinas de Boltzmann, RBMs y Deep Belief Networks.

2.  **Redes Neuronales y Backpropagation (Desde Cero y con PyTorch):**
    *   Fundamentos del cálculo de derivadas y la regla de la cadena para el backpropagation.
    *   Implementación manual de una neurona y operaciones (clase `Value` similar a micrograd) para entender el flujo de gradientes.
    *   Introducción a PyTorch y su uso para construir neuronas y Multi-Layer Perceptrons (MLP).
    *   Conceptos de función de pérdida (SSE), obtención de parámetros y descenso del gradiente.

3.  **Primeros Modelos de Lenguaje (Bigramas):**
    *   Preparación de datos y cálculo de probabilidades para modelos de bigramas estadísticos.
    *   Función de pérdida (likelihood, negative log likelihood).
    *   Implementación de bigramas utilizando redes neuronales (one-hot encoding, softmax).

4.  **Predicciones con MLP (Parte 1 y 2 - Inspirado en Bengio 2003):**
    *   Diseño de un MLP para la predicción de caracteres (capas de entrada con embeddings, capas ocultas, capas de salida).
    *   Proceso de entrenamiento: manejo de overfitting, división train/val/test, búsqueda de tasa de aprendizaje óptima.
    *   Mejoras en la red: optimización de la pérdida inicial, manejo de la saturación de `tanh`.
    *   Introducción e implementación de **Batch Normalization**.
    *   Implementación más profunda estilo PyTorch con clases personalizadas (`Linear`, `BatchNorm1d`, `Tanh`, `Sequential`).
    *   Análisis detallado del forward pass, backward pass y la distribución de gradientes.

5.  **WaveNet (CNNs para Secuencias):**
    *   Exploración de arquitecturas convolucionales (CNN) aplicadas a la generación de secuencias, basándose en conceptos de WaveNet.
    *   Implementación de la fusión jerárquica de características.
    *   Solución de problemas específicos de BatchNorm en arquitecturas más profundas y escalado del modelo.

6.  **GPT y Transformers (El Núcleo del Proyecto):**
    *   Preparación de datos para datasets más complejos (obras de Shakespeare, dataset de poesía en español).
    *   Construcción de un modelo GPT a nivel de carácter, paso a paso:
        *   Prerrequisitos y fundamentos de **Self-Attention** (promedios enmascarados, embeddings posicionales).
        *   Implementación de **Self-Attention** y **Multi-Head Attention**.
        *   Desarrollo de la arquitectura completa del **Transformer**:
            *   Bloques de Feed Forward.
            *   Agrupación en Bloques de Transformer.
            *   Conexiones Residuales.
            *   **Layer Normalization**.
            *   **Dropout**.
    *   Entrenamiento final del modelo Transformer con todos los componentes y evaluación de su capacidad generativa.

7.  **Manual de Usuario para Generación:**
    *   Instrucciones claras sobre cómo utilizar los modelos entrenados (Shakespeare y poesía) para generar nuevo texto, especificando cómo modificar parámetros como el dataset de entrada y el modelo cargado.

### Tecnologías y Conceptos Clave:

*   **Lenguaje y Librerías:** Python, PyTorch.
*   **Fundamentos de NN:** Backpropagation, Descenso de Gradiente, Funciones de Activación (tanh, ReLU, Softmax), Funciones de Pérdida (SSE, Cross-Entropy).
*   **Arquitecturas de Modelos:**
    *   Bigramas (Estadísticos y Neuronales)
    *   Multi-Layer Perceptron (MLP)
    *   Redes Convolucionales (aplicadas a secuencias - WaveNet)
    *   Transformers (GPT)
*   **Componentes Clave de Transformers:**
    *   Embeddings (de token y posicionales)
    *   Self-Attention
    *   Multi-Head Attention
    *   Conexiones Residuales
    *   Layer Normalization
    *   Dropout
*   **Técnicas de Entrenamiento y Regularización:**
    *   Batch Normalization
    *   Optimización de Hiperparámetros
    *   División de Datos (Train/Validation/Test)

### Objetivo:

Este proyecto busca ofrecer una comprensión práctica y teórica de la evolución de los modelos de lenguaje, implementando desde cero (o con un bajo nivel de abstracción usando PyTorch) los componentes esenciales de estas arquitecturas. El resultado final incluye modelos funcionales capaces de generar texto coherente a nivel de carácter.
