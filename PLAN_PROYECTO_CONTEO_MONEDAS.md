# Proyecto de Visión por Computadora — Conteo y clasificación de monedas

> **Documento de trabajo / bitácora de implementación**  
> Lenguaje: **C++** · Biblioteca: **OpenCV** · Entrada: `detectarMonedas.png` (2048 × 1594 píxeles)  
> **Objetivo de la sesión actual:** llegar al **hito H3 (50% funcional)** y detenerse. **No** implementar todavía el reconocimiento completo ni el conteo final.
>
> **Nota de implementación (2026-10-15):** el archivo real entregado es `assets/coins.png` y mide **2477 × 1929** (no 2048 × 1594). Tiene la misma composición escalada ×1.2095, así que todas las coordenadas se midieron de nuevo sobre esta resolución (ver H1).

## 0. Problema y alcance

La imagen original tiene dos zonas:
- **Izquierda:** monedas variadas sobre madera, algunas superpuestas o cortadas.
- **Derecha:** cuatro monedas estadounidenses de referencia: **penny (1¢), nickel (5¢), dime (10¢) y quarter (25¢)**.

**Meta final:** contar por separado cuántas monedas de cada tipo de referencia aparecen en la zona izquierda y marcar visualmente cuáles fueron reconocidas.

**Importante:** *detectar una circunferencia* no equivale a *reconocer una denominación*. Hay monedas no estadounidenses, diseños distintos y posibles caras opuestas. La salida deberá permitir **«desconocida / no reconocida»**, sin forzar la clasificación.

### Supuestos de partida

1. Se trabaja con la imagen proporcionada, no con video ni fotografías arbitrarias.
2. Las coordenadas de recorte pueden estar fijas al principio; deben validarse en pantalla.
3. El programa debe ser **explicable a partir de métodos vistos en clase**, sin redes neuronales ni sistemas entrenados externamente.
4. Usaremos Canny y Hough para localizar candidatos; SIFT para intentar reconocer tipos. Harris puede incorporarse solo como comparación opcional.
5. Cada moneda física debe contarse **una sola vez**, aunque produzca varios candidatos o varios matches.
6. Una moneda parcialmente tapada podría no detectarse con Hough; se documentará como limitación y, si hay tiempo, se estudiará una estrategia complementaria.

## 1. Estado del proyecto

**Convención:** `- [ ]` pendiente; `- [x]` implementado, ejecutado y verificado visualmente (no marcar solo por haber escrito código).

| Hito | Fase | Peso de avance | Estado | Evidencia mínima |
|---|---|---:|---|---|
| H0 | Preparación del proyecto | 10% | ☑ Verificado* | Compila y abre imagen |
| H1 | Recortes y referencias | 15% | ☑ Verificado | 5 imágenes independientes |
| H2 | Preprocesamiento y Canny | 15% | ☑ Verificado | Comparativa original / gris / suavizado / bordes |
| **H3** | **Círculos candidatos y revisión** | **10%** | **☑ Verificado** | **Monedas rodeadas, conteo provisional y errores identificados** |
| H4 | Extracción SIFT y descriptores | 20% | ☐ | Keypoints y descriptores por referencia/candidato |
| H5 | Correspondencia y clasificación | 20% | ☐ | Coincidencias razonadas + categoría «desconocida» |
| H6 | Conteo, validación y entrega | 10% | ☐ | Conteo por tipo + imagen final + reporte de errores |
| | **Total** | **100%** | | |

> **CORTE DE HOY: H0 + H1 + H2 + H3 = 50%** del plan por hitos. Es una medida de avance acordada, **no** una afirmación de que el algoritmo completo ya funciona al 50% ni de precisión de conteo.

\* H0: `imshow`/`waitKey` está implementado (opción `--show`) y el proceso queda esperando la tecla sin errores (backend Cocoa). Sin embargo, la ventana **no se confirmó visualmente**: la sesión no tenía permiso de captura de pantalla. La validación de H0 usa la evidencia alternativa «registro de consola». Falta confirmarla ejecutando `coin_detector --show`.

**Actualización de progreso** (rellenar al terminar cada fase):

```text
Fecha:            2026-10-15
Último hito validado: H3
Avance acumulado: 50 % (por hitos; no es precisión de conteo)
Qué funciona:     CMake (presets debug/release) + OpenCV 5.0.0; carga y validación de la imagen;
                  5 recortes ROI verificados; gris → Gauss → Canny; HoughCircles con 17 candidatos,
                  0 duplicados, 0 falsos positivos sobre la madera.
Qué falla:        ~9 monedas parciales u ocultas no se detectan (lista en H3); el candidato 15 es
                  dudoso; el radio del candidato 3 se ajusta al borde interior.
Parámetros usados: Gauss 9×9 σ=2.0; Canny 60/120; Hough dp=1.2, minDist=100, param1=120,
                  param2=40, radio 95–220 (todos en include/params.hpp).
Captura o archivo de evidencia: output/01_search_region.png, output/08_canny_edges.png,
                  output/09_hough_circles.png, output/10_preprocessing_comparison.png + consola.
Siguiente paso:   H4 — SIFT con máscara circular sobre referencias y candidatos no cortados.
```

---

## 2. Organización del proyecto

```text
conteo-monedas/
├── README.md                         # Este documento
├── CMakeLists.txt                    # Si se compila con CMake
├── assets/
│   └── detectarMonedas.png
├── src/
│   └── main.cpp
├── outputs/
│   ├── fase1/
│   ├── fase2/
│   ├── fase3/
│   └── final/
└── notas/
    └── bitacora.md
```

**Estructura real implementada** (adaptada; ver `README.md`): `include/` + `src/` (un archivo por fase: `regions.cpp` H1, `preprocessing.cpp` H2, `circle_detection.cpp` H3), `include/params.hpp` con todas las constantes, `CMakePresets.json` y una sola carpeta `output/` con archivos numerados en vez de `outputs/fase*/`:

| Plan original | Archivo generado |
|---|---|
| `outputs/fase1/busqueda.png` | `output/01_search_region.png` |
| `outputs/fase1/` (referencias) | `output/02_reference_penny.png` … `05_reference_quarter.png` |
| `outputs/fase2/canny.png` | `output/08_canny_edges.png` (+ `06_grayscale`, `07_gaussian`, `10_preprocessing_comparison`) |
| `outputs/fase3/candidatos_hough.png` | `output/09_hough_circles.png` |

**Buenas prácticas:** conservar el original intacto; guardar resultados por fase; trabajar con `cv::Mat`, `cv::Rect`, `cv::imshow`, `cv::imwrite`; almacenar parámetros en constantes al inicio; revisar todas las etapas en ventanas y archivos. Si no se usa CMake, mantener igualmente los directorios y configurar OpenCV en el IDE.

---

# BLOQUE A — IMPLEMENTAR EN ESTA SESIÓN (50%)

## H0 — Configuración y carga (10%)

**Objetivo:** tener un ejecutable reproducible y cargar correctamente la imagen.

**Tareas**
- [x] Crear carpetas del proyecto y colocar la imagen en `assets/` (como `assets/coins.png`).
- [x] Configurar OpenCV y compilar un `main.cpp` mínimo (CMake + presets `debug`/`release`, 0 warnings con `-Wall -Wextra -Wpedantic`).
- [x] Cargar con `cv::imread(..., IMREAD_COLOR)`. La ruta por defecto es absoluta, fijada por CMake, así que funciona desde cualquier directorio; se puede cambiar con `--image`.
- [x] Verificar `image.empty()` y mostrar dimensiones (`cols`, `rows`). Probado: archivo inexistente, archivo no decodificable y resolución distinta (error explicativo, código de salida 1).
- [ ] Visualizar con `imshow()` y cerrar limpiamente con `waitKey()`. Está implementado con `--show`, pero la ventana falta confirmarla en pantalla (ver nota * arriba).

**Validación:** imagen visible, dimensiones esperadas **2048 × 1594**, sin errores de ruta ni enlace de librerías.
**Resultado:** imagen de **2477 × 1929**, 3 canales, 8 bits; el alfa del PNG es 255 en todos los píxeles, así que se descarta sin pérdida. No hay errores de ruta ni de enlace.

**Evidencia:** captura de la ventana original o registro de consola.  
**Avance acumulado al verificar:** **10%**.

## H1 — Separación de regiones y referencias (15%)

**Objetivo:** aislar la zona izquierda y cuatro monedas de la zona derecha.

**Tareas**
- [x] Definir `Rect(x, y, ancho, alto)` para la zona de búsqueda.
- [x] Definir cuatro `Rect` para penny, nickel, dime y quarter.
- [x] Recortar cada ROI con `imagen(rect).clone()`. Se probó que modificar un recorte no altera el original.
- [x] Guardar la búsqueda y las cuatro referencias (`output/01…05_*.png`, más `00_roi_overview.png`).
- [x] Abrir cada recorte y confirmar que incluye la moneda correcta. Cada uno tiene la moneda completa y ningún trozo de las vecinas (comprobado también por píxeles).
- [x] Verificar que los rectángulos estén completamente dentro de los límites de la imagen (`isInsideImage`; probado con rectángulos fuera de rango, negativos y vacíos).

**Coordenadas iniciales propuestas — revisar antes de considerarlas correctas:**

| ROI | `Rect(x, y, w, h)` |
|---|---|
| Búsqueda | `(86, 106, 880, 1317)` |
| Penny | `(1315, 219, 178, 213)` |
| Nickel | `(1514, 207, 206, 251)` |
| Dime | `(1445, 444, 215, 230)` |
| Quarter | `(1683, 402, 276, 342)` |

Estas coordenadas dependen **exactamente** de la resolución original. Algunos recortes pueden contener fondo u otra moneda; afinarlos manualmente si es necesario. **No** recortar usando posiciones de ventanas escaladas en pantalla.

**Coordenadas verificadas sobre `coins.png` (2477 × 1929)** — en `include/params.hpp`:

| ROI | Propuesta × 1.2095 | `Rect` final | Motivo |
|---|---|---|---|
| Búsqueda | (104, 128, 1064, 1593) | `(104, 128, 1064, 1595)` | Justo dentro del marco oscuro (columnas 92–98 / 1169–1175, fila 126–127; la foto termina en la fila 1722). |
| Penny | (1590, 265, 215, 258) | `(1588, 263, 219, 264)` | Margen de ~5 px alrededor de la moneda. |
| Nickel | (1831, 250, 249, 304) | `(1829, 253, 245, 297)` | Termina antes del borde superior del dime. |
| Dime | (1748, 537, 260, 278) | `(1757, 547, 250, 297)` | **La propuesta cortaba ~20 px de la parte inferior del dime.** |
| Quarter | (2036, 486, 334, 414) | `(2042, 497, 332, 397)` | Empieza en x = 2042 para excluir el borde del nickel (llega a x = 2041). |

Observación para H4: las monedas de referencia están **estiradas verticalmente** (elipses de ~205 × 250 px para el penny), mientras que las de la foto son redondas.

**Validación:** cinco imágenes guardadas y coherentes con las regiones previstas. ✔
**Evidencia:** `output/00_roi_overview.png`, `output/01…05_*.png`.  
**Avance acumulado:** **25%**.

## H2 — Escala de grises, suavizado y Canny (15%)

**Objetivo:** producir un mapa de bordes razonable para localizar monedas, sin considerar todavía sus tipos.

**Tareas**
- [x] Convertir **solo la ROI izquierda** con `cvtColor(..., COLOR_BGR2GRAY)`.
- [x] Aplicar `GaussianBlur` para reducir ruido y textura pequeña.
- [x] Aplicar `Canny` con dos umbrales y guardar la imagen de bordes.
- [x] Comparar lado a lado: color, grises, suavizada y bordes (`output/10_preprocessing_comparison.png`).
- [x] Ajustar los umbrales y guardar los valores usados (tabla siguiente y `include/params.hpp`).
- [x] Confirmar que los límites de varias monedas son visibles, sin exagerar la textura de madera.

**Pruebas H2 (2026-10-15)** — porcentaje de píxeles de borde en la ROI de búsqueda:

| Gauss | Canny bajo/alto | Bordes | Observación |
|---|---|---:|---|
| 5×5, σ 1.5 (inicio) | 50 / 150 | 5.6 % | Mucha veta de madera y grabados internos de las monedas. |
| 9×9, σ 2.0 | 50 / 150 | 1.6 % | Limpio, pero algunos contornos quedan cortados. |
| **9×9, σ 2.0** | **60 / 120** | **2.1 %** | **Elegido.** Contornos casi cerrados y pocas líneas de veta. |
| 13×13, σ 3.0 | 60 / 120 | 0.8 % | Los arcos empiezan a romperse. |
| 17×17, σ 4.0 | 40 / 100 | 0.5 % | Se pierden partes de contornos. |

Decisión: bajo = alto / 2 porque `HoughCircles` (HOUGH_GRADIENT) aplica internamente `Canny(param1/2, param1)`. Con `param1 = 120`, `canny.png` muestra los mismos bordes con los que vota Hough.

**Valores de arranque (no definitivos):**

```cpp
GaussianBlur(gray, blurred, Size(5, 5), 1.5);
Canny(blurred, edges, 50, 150);
```

**Por qué:** en los apuntes de Canny el procesamiento pasa por reducción de ruido, gradientes, supresión de no máximos, doble umbral e histéresis. `cv::Canny` ejecuta la detección; el suavizado explícito permite visualizar y controlar la preparación de la entrada.

**Riesgos:** la veta de madera, letras grabadas en monedas, sombras y reflejos también producen bordes. No exigir un contorno perfecto en esta fase.

**Validación:** archivo `outputs/fase2/canny.png` donde se distingan suficientes bordes circulares para experimentar con Hough. ✔ (`output/08_canny_edges.png`)  
**Evidencia:** comparativa visual y parámetros anotados.  
**Avance acumulado:** **40%**.

## H3 — Hough para círculos + revisión de candidatos (10%) — **ALTO DE HOY**

**Objetivo:** localizar círculos candidatos `(x, y, radio)` y dibujarlos sobre el recorte izquierdo.

**Tareas**
- [x] Aplicar `HoughCircles` sobre la imagen gris suavizada.
- [x] Probar parámetros de distancia mínima entre centros y rango de radios.
- [x] Dibujar círculo, centro e id numerado para cada candidato sobre una **copia a color**.
- [x] Guardar la imagen con candidatos (`output/09_hough_circles.png`). Mostrarla con `--show` falta confirmarlo en pantalla, igual que en H0.
- [x] Imprimir en consola cantidad de **círculos candidatos** (no monedas de cada clase), con una tabla id / centro / radio / cortado por el borde.
- [ ] Marcar visualmente **falsos positivos, duplicados y monedas omitidas**. Por ahora está registrado por escrito (id y coordenadas, abajo). La anotación sobre la imagen se hará con la verdad de referencia de H6.
- [x] Documentar parámetros y elegir una configuración reproducible.

**Pruebas H3 (2026-10-15)** — `HOUGH_GRADIENT`, param1 = 120, sobre `07_gaussian.png`. Se compararon con una lista manual de unas 22 monedas visibles total o parcialmente:

| dp | minDist | param2 | Radio | Círculos | Falsos positivos | Duplicados |
|---|---:|---:|---|---:|---:|---:|
| 1.2 | 50 | 30 | 42–194 (propuesta) | 79 | muchos | muchos |
| 1.2 | 100 | 30 | 95–220 | 24 | 4 | 0 |
| 1.2 | 100 | 32 | 95–220 | 21 | 1 | 0 |
| 1.2 | 100 | 35 | 95–220 | 18 | 0 | 0 |
| **1.2** | **100** | **40** | **95–220** | **17** | **0** | **0** |
| 1.2 | 100 | 45–50 | 95–220 | 15 | 0 | 0 |
| 1.2 | 100 | 55–60 | 95–220 | 14 | 0 | 0 |
| 1.2 | 60 | 35 | 95–220 | 23 | 0 | 4 |
| 1.2 | 80–150 | 35 | 95–220 | 18–19 | 0 | 0 |
| 1.2 | 100 | 35 | 80–240 | 23 | 3 | 0 |
| 1.5 | 100 | 45 | 95–220 | 20 | 1 | 0 |
| 2.0 | 100 | 45 | 95–220 | 50 | 29 | 0 |

Los falsos positivos y duplicados de la tabla se contaron emparejando automáticamente cada centro con la lista manual (tolerancia de 70 px). Las configuraciones con param2 = 35 y param2 = 40 se revisaron además a simple vista.

Decisión: **param2 = 40**. Hay 0 falsos positivos a partir de 33; 40 deja margen y solo pierde una moneda parcial respecto a 35. `minDist = 100` está en el centro del rango estable (80–150). Además, `selectCandidates` descarta explícitamente un círculo cuyo centro esté a menos de 0.5 × radio de otro más votado; con esta configuración no eliminó ninguno.

**Revisión de los 17 candidatos** (ids de `09_hough_circles.png`; coordenadas dentro de la ROI de búsqueda):

- **Bien ajustados (16):** 1 (5 francos), 2 (one penny británico), 3 (Chipre), 4 (25 neerlandés), 5, 6 (Austria), 7 (20 cent euro), 8, 9, 10 (penny EE. UU.), 11 (penny EE. UU.), 12, 14 (Elizabeth II), y los parciales en el borde 13, 16 y 17 (5 cent euro).
- **Dudoso (1):** 15 sigue el borde superior e izquierdo de una moneda dorada cuya mitad inferior está tapada. Es plausible, pero no se puede confirmar a simple vista.
- **Radio impreciso:** 3 se ajusta al anillo interior del borde (radio ~12–15 px menor).
- **Falsos positivos:** 0. **Duplicados:** 0.
- **Omitidas (~9, todas parciales u ocultas):** plateada «…REICH 1988» detrás del 17 (~1010, 360); plateada casi oculta bajo 10 y 14 (~400, 900); dorada «1963» encima de la del 15 (~1010, 1290); plateada en el borde derecho (~1015, 1405); plateada en la esquina inferior derecha (~1000, 1530); franja plateada en el borde derecho (~1060, 930); y tres fragmentos en el borde inferior (~150, ~640 y ~830 en x; y ≈ 1590).

**Conteo provisional: 17 candidatos geométricos.** No es el número de monedas estadounidenses.

**Punto de partida, sujeto a ajuste visual:**

```cpp
vector<Vec3f> circles;
HoughCircles(
    blurred, circles, HOUGH_GRADIENT,
    1.2, 50, 120, 30, 35, 160
);
```

Significado de los parámetros clave:
- `dp`: resolución relativa del acumulador de centros.
- `minDist`: distancia mínima entre centros detectados.
- `param1`: umbral alto interno de Canny para el método de gradiente.
- `param2`: umbral de aceptación de candidatos a centro.
- `minRadius` / `maxRadius`: rango de tamaños posibles.

**Nota técnica:** OpenCV `HoughCircles` recibe normalmente una **imagen de intensidades**, no el mapa binario Canny generado por nosotros: el algoritmo de gradiente utiliza detección de bordes internamente. Conservamos `canny.png` para comprender y ajustar visualmente el proceso.

**Validación:** `outputs/fase3/candidatos_hough.png` con círculos superpuestos y un registro de aciertos/errores. ✔ (`output/09_hough_circles.png` + revisión anterior) No es necesario detectar todas las monedas para declarar terminado el experimento: sí debe funcionar el flujo, y debemos explicar qué no detecta.

**Evidencia para entregar o demostrar hoy:**
- `busqueda.png`.
- `canny.png`.
- `candidatos_hough.png`.
- Capturas o bitácora de parámetros.
- Número provisional de **candidatos** con observaciones de errores.

**Avance acumulado:** **50%**.  
**DETENER IMPLEMENTACIÓN AQUÍ.** No activar clasificación ni afirmar conteos por denominación.

---

# BLOQUE B — PRÓXIMAS SESIONES (50% RESTANTE)

## H4 — SIFT: puntos clave y descriptores (20%)

**Objetivo:** generar descriptores para cada referencia y para cada moneda candidata.

**Plan**
- [ ] Convertir las referencias a grises para SIFT; conservar copias a color.
- [ ] Enmascarar la zona circular de cada referencia y candidato para evitar que la madera o el fondo dominen los puntos clave.
- [ ] Si hace falta, normalizar tamaño de parches (sin deformar monedas).
- [ ] Crear `SIFT::create()` y usar `detectAndCompute()`.
- [ ] Guardar visualizaciones de keypoints en referencia y candidatos.
- [ ] Comprobar que los descriptores no estén vacíos y registrar sus dimensiones.

**Concepto visto en clase:** SIFT detecta extremos de DoG en un espacio de escalas, refina puntos, asigna orientación y construye un descriptor clásico `4 × 4 × 8 = 128` valores. Es robusto a rotación y escala, pero no reconoce por sí solo el nombre de una moneda.

**Criterio:** descriptores SIFT disponibles para las cuatro referencias y para candidatos válidos.

## H5 — Matching y clasificación por tipo (20%)

**Objetivo:** decidir si cada candidato coincide de forma fiable con alguna de las cuatro referencias.

**Plan**
- [ ] Comparar descriptores con `BFMatcher(NORM_L2)`.
- [ ] Aplicar prueba de razón (*Lowe ratio test*) para rechazar matches ambiguos.
- [ ] Analizar cuántas correspondencias válidas hay por referencia.
- [ ] Cuando existan suficientes correspondencias, comprobar consistencia geométrica con RANSAC.
- [ ] Definir un umbral de aceptación basándose en pruebas, no en un número arbitrario de coincidencias aisladas.
- [ ] Etiquetar `penny`, `nickel`, `dime`, `quarter` o `desconocida`.
- [ ] En caso de varias clases plausibles, conservar `desconocida` o `ambigua` hasta mejorar validación.

**Limitación que hay que probar:** las referencias muestran una sola cara y un diseño concreto. Variaciones de año, cara opuesta, iluminación u oclusión pueden hacer fallar SIFT. También conviene comprobar que las cuatro clases de referencia sean suficientes para el objetivo del profesor.

**Criterio:** ejemplos verificables de matches verdaderos y falsos, con un rechazo explícito de candidatos dudosos.

## H6 — Conteo, evaluación y entrega (10%)

**Objetivo:** reportar cantidades por tipo sin doble conteo.

**Plan**
- [ ] Deduplicar círculos solapados o centros demasiado cercanos que representen la misma moneda.
- [ ] Contar cada objeto aceptado **una sola vez**.
- [ ] Mostrar y guardar la imagen final con círculos y etiquetas.
- [ ] Mostrar tabla de conteos por denominación y cantidad de no identificadas.
- [ ] Comparar con anotación manual de verdad de referencia (*ground truth*).
- [ ] Separar errores de **detección** de errores de **clasificación**.
- [ ] Explicar casos no resolubles por oclusión, reverso o falta de características.

**Salida esperada (estructura, no resultados reales):**

```text
Penny:        [cantidad]
Nickel:       [cantidad]
Dime:         [cantidad]
Quarter:      [cantidad]
Desconocidas: [cantidad]
Total reconocidas: [cantidad]
```

**Criterio:** reporte final reproducible con evidencias visuales, parámetros utilizados y limitaciones conocidas.

---

## 3. Métodos del curso y relación con la solución

| Método visto | Papel real en este proyecto | Fuente de clase |
|---|---|---|
| **Sobel / gradientes** | Explican los cambios de intensidad que utiliza la detección de bordes | `Detector_Canny_Paso_a_Paso_ES.docx` y material de Harris |
| **Canny** | Bordes finos; análisis visual para preparar detección geométrica | `Detector_Canny_Paso_a_Paso_ES.docx` |
| **Hough** | Votación de centros y radios; máximos y supresión de duplicados | `Entendiendo la Transformada de Hough.pdf` |
| **SIFT** | Puntos clave, orientación, escala y descriptores para comparar monedas | `Detector SIFT_soloTextoCorrected.pdf` y `ManualSIFT.cpp` |
| **Harris** | Detección de esquinas; útil como demostración/comparación, **no** esencial para el conteo | `ExplicacionHarrisEstudiantes.pdf` |

### Código proporcionado que se puede reutilizar

- `InteractiveDetector (1).cpp`: referencia **directa** para Canny, `HoughCircles`, `SIFT::create`, `detectAndCompute` y controles de parámetros. Es el punto de partida más práctico.
- `Hough.cpp` y `hough-(C++).txt`: muestran acumuladores y supresión de máximos para **líneas**. Útiles para entender el principio de votación, **no** copiar directamente como detector de círculos.
- `ManualSIFT.cpp`: implementación didáctica de pirámides Gaussianas, DoG, extremos, orientación y descriptor; sirve para justificar cómo funciona SIFT. No es necesario reconstruirlo desde cero para el MVP.
- `harris-detector-code-(C++).txt`: gradientes, tensor de estructura y respuesta Harris; fuente de comparación opcional.
- `QUIZ-SIFT-DETECTOR.txt`: repaso de invariancia, descriptor de 128 elementos y normalización.

### Terminología que debe respetarse en el reporte

- **Región de interés (ROI):** recorte espacial relevante.
- **Gradiente:** magnitud y dirección del cambio de intensidad.
- **Canny:** suavizado, gradiente, supresión de no máximos, doble umbral e histéresis.
- **Hough para círculos:** parámetros `(centro_x, centro_y, radio)`, acumulador, votación y máximos.
- **SIFT:** detector **y** descriptor; un *keypoint* no equivale a un objeto.
- **Matching:** comparación de descriptores y rechazo de correspondencias ambiguas.

---

## 4. Cómo trabajar sin perder el control del avance

1. **Implementar una fase a la vez** y ejecutar antes de agregar la siguiente.
2. **Guardar imágenes intermedias** para inspeccionar los resultados.
3. **Anotar ajustes de parámetros** después de cada prueba y conservar al menos una configuración que funcione.
4. **Marcar `[x]` únicamente al superar la validación**; no marcar tareas que solo se planearon.
5. **Registrar bloqueos concretos:** por ejemplo, «Hough detecta textura de madera como círculos» es útil; «no funciona» no ayuda a depurar.
6. **Parar después de H3**. El resto queda especificado para poder continuar más adelante sin replanificar desde cero.

### Plantilla de bitácora para una prueba

```markdown
### Prueba [número] — [fecha]
- Fase/hito:
- Cambio realizado:
- Parámetros:
- Imagen generada:
- Resultado observado:
- Errores y posibles causas:
- Decisión: mantener / revertir / ajustar
```

## 5. Próxima acción concreta

**Actualización 2026-10-15:** H0–H3 están implementados y verificados (ver estado en la sección 1). La próxima acción es **H4**: añadir el módulo `features` de OpenCV 5 a CMake y extraer SIFT con máscara circular de las cuatro referencias y de los candidatos no cortados por el borde.

*Texto original:* **Empezar por H0 y H1** en `src/main.cpp`: cargar la imagen, verificar el tamaño, recortar izquierda + cuatro referencias, y comprobar los cinco archivos guardados. Solo después agregar el preprocesamiento H2 y la detección geométrica H3.

**Definición de «la mitad del proyecto, por hoy»:** contar con una tubería ejecutable **imagen → ROI → grises → suavizado → Canny → candidatos Hough**, pruebas visuales guardadas y problemas conocidos escritos. **No** significa que ya podamos contar correctamente pennies, nickels, dimes y quarters: eso empieza en H4–H6.
