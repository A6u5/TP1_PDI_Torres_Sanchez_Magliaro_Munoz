import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(
    r"C:\Users\facum\OneDrive\Documentos\2.T.U.I.A\Procesamineto Imagenes\TP_1\TP1_PDI_Torres_Sanchez_Magliaro\TP_2\monedas.jpg", cv2.IMREAD_GRAYSCALE)


#SI se intenta un umbralado global como se hizo en otras ocaciones no funciona ya que el resultado confunde mas que ayudar

"""
img_th = img < 128
plt.imshow(img_th, cmap="gray")
plt.title("Imagen Umbralada")
plt.axis("off")
plt.tight_layout()
plt.show()

cv2.IMREAD_GRAYSCALE

#HISTOGRAMA DE INTENSIDADES

plt.figure(figsize=(8,4))
plt.hist(img_gray.ravel(), bins=256, range=[0,256])
plt.title("Histograma de intensidades")
plt.xlabel("Intensidad")
plt.ylabel("Cantidad de píxeles")
plt.show()
"""

#### ITEM A) ####

img = cv2.resize(img, None, fx=0.65, fy=0.65)

# Hago la imagen suave y aplico canny 
img_suave = cv2.GaussianBlur(img, (5, 5), 1)

img_canny = cv2.Canny(img_suave, 45, 100)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

img_cierre = cv2.morphologyEx(
    img_canny, cv2.MORPH_CLOSE, kernel
)

contornos, jerarquia = cv2.findContours(
    img_cierre,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
# Cuento cuanta cantidad de contornos detecto
print("Cantidad total de contornos detectados:", len(contornos))

# Filtro contornos por área, los mas pequeños son ruido, los mas grandes objetos (dados monedas)
area_min = 500

contornos_validos = []

for cnt in contornos:
    area = cv2.contourArea(cnt)

    if area > area_min:
        contornos_validos.append(cnt)

print("Objetos detectados:", len(contornos_validos))


#Hago la mascara. 
mascara_objetos = np.zeros_like(img)

for cnt in contornos_validos:

    hull = cv2.convexHull(cnt)

    cv2.drawContours(
        mascara_objetos,
        [hull],
        -1,
        255,
        -1
    )


# Dibujamos los contornos sobre la imagen original para verlos en comparacion (Esto es opcional pero queda lindo)
img_contornos = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.drawContours(
    img_contornos,
    contornos_validos,
    -1,
    (255, 0, 0),
    2
)


### TODOS LOS PLOTS ####

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_cierre, cmap="gray")
plt.title("Cierre morfológico")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title("Contornos válidos")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mascara_objetos, cmap="gray")
plt.title("Mascara")
plt.axis("off")

plt.tight_layout()
plt.show()


####### ITEM B ##########

monedas = []
dados = []

for cnt in contornos_validos:

    hull = cv2.convexHull(cnt)

    area = cv2.contourArea(hull)
    perimetro = cv2.arcLength(hull, True)

    fp = area / (perimetro ** 2)

    if fp > 0.075:
        monedas.append(hull)
    else:
        dados.append(hull)

print("Monedas detectadas:", len(monedas))
print("Dados detectados:", len(dados))

img_clasificada = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img_clasificada, monedas, -1, (0, 255, 0), 3)
cv2.drawContours(img_clasificada, dados, -1, (0, 0, 255), 3)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_clasificada, cv2.COLOR_BGR2RGB))
plt.title("Monedas (verde) - Dados (rojo)")
plt.axis("off")
plt.show()